use std::sync::{atomic::AtomicI32, Arc, Mutex};

use crate::{
    camera::RealisticCamera,
    geometry::{abs_dot3, dot3, Bounds3f, Point3f, Point3i, RayDifferential, Vector3f},
    interaction::{Interaction, MediumInteraction, SurfaceInteraction},
    misc::{clamp_t, AtomicF64},
    primitives::Primitive,
    reflection::{Bsdf, BXDF_ALL, BXDF_SPECULAR, BXDF_TRANSMISSION},
    samplers::{halton::Halton, GlobalSampler, Sampler, StartPixel},
    sampling::Distribution1D,
    scene::Scene,
    spectrum::{ISpectrum, Spectrum},
    SPECTRUM_N,
};

use super::*;

use rayon::prelude::*;

// SPPM Declarations
// class SPPMIntegrator : public Integrator {
//   public:
//     // SPPMIntegrator Public Methods
//     SPPMIntegrator(std::shared_ptr<const Camera> &camera, int nIterations,
//                    int photonsPerIteration, int maxDepth,
//                    Float initialSearchRadius, int writeFrequency)
//         : camera(camera),
//           initialSearchRadius(initialSearchRadius),
//           nIterations(nIterations),
//           maxDepth(maxDepth),
//           photonsPerIteration(photonsPerIteration > 0
//                                   ? photonsPerIteration
//                                   : camera->film->croppedPixelBounds.Area()),
//           writeFrequency(writeFrequency) {}
//     void Render(const Scene &scene);

//   private:
//     // SPPMIntegrator Private Data
//     std::shared_ptr<const Camera> camera;
//     const Float initialSearchRadius;
//     const int nIterations;
//     const int maxDepth;
//     const int photonsPerIteration;
//     const int writeFrequency;
// };

#[derive(Debug)]
pub struct SPPMIntegrator {
    cam: Arc<RealisticCamera>,

    init_search_radius: f64,
    n_iters: usize,
    max_depth: usize,
    photons_per_iter: usize,
    write_freq: usize,

    light_distr: Arc<Distribution1D>,
}

#[derive(Debug, Default, Clone)]
pub struct VisiblePoint {
    pub p: Point3f,
    pub wo: Vector3f,
    pub bsdf: Option<Bsdf>,
    pub beta: Spectrum<SPECTRUM_N>,
}

#[derive(Debug, Default)]
struct SPPMPixel {
    pub radius: f64,

    pub ld: Spectrum<SPECTRUM_N>,
    pub vp: VisiblePoint,
    pub phi: [AtomicF64; SPECTRUM_N],
    pub m: AtomicI32,
    pub n: f64,
    pub tau: Spectrum<SPECTRUM_N>,
}

struct SPPMPixelListNode {
    pixel: SPPMPixel,

    next: Box<SPPMPixelListNode>,
}

pub fn to_grid(p: &Point3f, bounds: &Bounds3f, grid_res: [i64; 3], pi: &mut Point3i) -> bool {
    let mut in_bounds = true;
    let pg = bounds.offset(p);
    for i in 0..3 {
        pi[i] = grid_res[i as usize] * pg[i] as i64;
        in_bounds &= pi[i] >= 0 && pi[i] < grid_res[i as usize];
        pi[i] = clamp_t(pi[i], 0, grid_res[i as usize] - 1);
    }
    in_bounds
}

fn hash(p: &Point3f, hash_size: usize) -> usize {
    ((p.x * 73856093.0) as usize ^ (p.y * 19349663.0) as usize ^ (p.z * 83492791.0) as usize)
        % hash_size
}

impl Integrator for SPPMIntegrator {
    fn render(&mut self, scene: &Scene) {
        // Initialize _pixelBounds_ and _pixels_ array for SPPM
        let pixel_bounds = self.cam.camera.film.cropped_pixel_bounds;
        let n_pixels = pixel_bounds.area();
        let mut tmp_pixels = vec![];
        for _pidx in 0..n_pixels as usize {
            let mut tmp = SPPMPixel::default();
            tmp.radius = self.init_search_radius;
            tmp_pixels.push(tmp);
        }
        let am_pixels = Arc::new(Mutex::new(tmp_pixels));
        let inv_sqrt_spp = 1.0 / (self.n_iters as f64).sqrt();

        if scene.lights.len() == 0 {
            self.light_distr = Arc::new(Distribution1D::default());
        } else {
            let mut light_power = vec![];
            for light in &scene.lights {
                light_power.push(light.power().y());
            }
            self.light_distr = Arc::new(Distribution1D::new(light_power));
        }

        // Perform _nIterations_ of SPPM integration
        let mut halton = Halton::new(&pixel_bounds, true);
        let mut halton_sampler =
            Arc::new(Mutex::new(GlobalSampler::new(self.n_iters as u64, halton)));

        // Compute number of tiles to use for SPPM camera pass
        let pixel_extent = pixel_bounds.diagonal();
        let tile_size = 16;
        let n_tiles_x = ((pixel_extent.x + tile_size - 1) / tile_size) as usize;
        let n_tiles_y = ((pixel_extent.y + tile_size - 1) / tile_size) as usize;

        for iter in 0..self.n_iters {
            // Generate SPPM visible points
            (0..n_tiles_x).into_par_iter().for_each(|tile_x| {
                (0..n_tiles_y).into_par_iter().for_each(|tile_y| {
                    // let mut pixels = am_pixels.lock().unwrap();
                    // Follow camera paths for _tile_ in image for SPPM
                    let tile_idx = tile_y * n_tiles_x + tile_x;
                    let mut tile_sampler = halton_sampler.lock().unwrap();

                    // Compute _tileBounds_ for SPPM tile
                    let x0 = pixel_bounds.p_min.x + tile_x as i64 * tile_size;
                    let x1 = (x0 + tile_size).min(pixel_bounds.p_max.x);
                    let y0 = pixel_bounds.p_min.y + tile_y as i64 * tile_size;
                    let y1 = (y0 + tile_size).min(pixel_bounds.p_max.y);
                    let tile_bounds = Bounds2i::new(Point2i::new(x0, y0), Point2i::new(x1, y1));
                    for p_pixel in tile_bounds.into_iter() {
                        // Prepare _tileSampler_ for _pPixel_
                        tile_sampler.start_pixel(p_pixel);
                        tile_sampler.set_sample_number(iter as u64);
                        // Generate camera ray for pixel for SPPM
                        let camera_sample = tile_sampler.get_camerasample(&p_pixel);
                        let mut ray = RayDifferential::default();
                        let beta = Spectrum::<SPECTRUM_N>::from(
                            self.cam.generate_ray_differential(&camera_sample, &mut ray),
                        );
                        if beta.is_black() {
                            continue;
                        }
                        ray.scale_differentials(inv_sqrt_spp);

                        // Follow camera ray path until a visible point is created

                        // Get _SPPMPixel_ for _pPixel_
                        let p_pixel_o = p_pixel - pixel_bounds.p_min;
                        // int pixelOffset =
                        //     pPixelO.x +
                        //     pPixelO.y * (pixelBounds.pMax.x - pixelBounds.pMin.x);
                        let pixel_offset = p_pixel_o.x
                            + p_pixel_o.y * (pixel_bounds.p_max.x - pixel_bounds.p_min.x);
                        // SPPMPixel &pixel = pixels[pixelOffset];
                        // let pixel = [pixel_offset as usize];
                        // bool specularBounce = false;
                        // for (int depth = 0; depth < maxDepth; ++depth) {
                        //     SurfaceInteraction isect;
                        //     ++totalPhotonSurfaceInteractions;
                        //     if (!scene.Intersect(ray, &isect)) {
                        //         // Accumulate light contributions for ray with no
                        //         // intersection
                        //         for (const auto &light : scene.lights)
                        //             pixel.Ld += beta * light->Le(ray);
                        //         break;
                        //     }
                        //     // Process SPPM camera ray intersection

                        //     // Compute BSDF at SPPM camera ray intersection
                        //     isect.ComputeScatteringFunctions(ray, arena, true);
                        //     if (!isect.bsdf) {
                        //         ray = isect.SpawnRay(ray.d);
                        //         --depth;
                        //         continue;
                        //     }
                        //     const BSDF &bsdf = *isect.bsdf;

                        //     // Accumulate direct illumination at SPPM camera ray
                        //     // intersection
                        //     Vector3f wo = -ray.d;
                        //     if (depth == 0 || specularBounce)
                        //         pixel.Ld += beta * isect.Le(wo);
                        //     pixel.Ld +=
                        //         beta * UniformSampleOneLight(isect, scene, arena,
                        //                                      *tileSampler);

                        //     // Possibly create visible point and end camera path
                        //     bool isDiffuse = bsdf.NumComponents(BxDFType(
                        //                          BSDF_DIFFUSE | BSDF_REFLECTION |
                        //                          BSDF_TRANSMISSION)) > 0;
                        //     bool isGlossy = bsdf.NumComponents(BxDFType(
                        //                         BSDF_GLOSSY | BSDF_REFLECTION |
                        //                         BSDF_TRANSMISSION)) > 0;
                        //     if (isDiffuse || (isGlossy && depth == maxDepth - 1)) {
                        //         pixel.vp = {isect.p, wo, &bsdf, beta};
                        //         break;
                        //     }

                        //     // Spawn ray from SPPM camera path vertex
                        //     if (depth < maxDepth - 1) {
                        //         Float pdf;
                        //         Vector3f wi;
                        //         BxDFType type;
                        //         Spectrum f =
                        //             bsdf.Sample_f(wo, &wi, tileSampler->Get2D(),
                        //                           &pdf, BSDF_ALL, &type);
                        //         if (pdf == 0. || f.IsBlack()) break;
                        //         specularBounce = (type & BSDF_SPECULAR) != 0;
                        //         beta *= f * AbsDot(wi, isect.shading.n) / pdf;
                        //         if (beta.y() < 0.25) {
                        //             Float continueProb =
                        //                 std::min((Float)1, beta.y());
                        //             if (tileSampler->Get1D() > continueProb) break;
                        //             beta /= continueProb;
                        //         }
                        //         ray = (RayDifferential)isect.SpawnRay(wi);
                        //     }
                        // }
                    }
                })
            })

            //     // Create grid of all SPPM visible points
            //     int gridRes[3];
            //     Bounds3f gridBounds;
            //     // Allocate grid for SPPM visible points
            //     const int hashSize = nPixels;
            //     std::vector<std::atomic<SPPMPixelListNode *>> grid(hashSize);
            //     {
            //         ProfilePhase _(Prof::SPPMGridConstruction);

            //         // Compute grid bounds for SPPM visible points
            //         Float maxRadius = 0.;
            //         for (int i = 0; i < nPixels; ++i) {
            //             const SPPMPixel &pixel = pixels[i];
            //             if (pixel.vp.beta.IsBlack()) continue;
            //             Bounds3f vpBound = Expand(Bounds3f(pixel.vp.p), pixel.radius);
            //             gridBounds = Union(gridBounds, vpBound);
            //             maxRadius = std::max(maxRadius, pixel.radius);
            //         }

            //         // Compute resolution of SPPM grid in each dimension
            //         Vector3f diag = gridBounds.Diagonal();
            //         Float maxDiag = MaxComponent(diag);
            //         int baseGridRes = (int)(maxDiag / maxRadius);
            //         CHECK_GT(baseGridRes, 0);
            //         for (int i = 0; i < 3; ++i)
            //             gridRes[i] = std::max((int)(baseGridRes * diag[i] / maxDiag), 1);

            //         // Add visible points to SPPM grid
            //         ParallelFor([&](int pixelIndex) {
            //             MemoryArena &arena = perThreadArenas[ThreadIndex];
            //             SPPMPixel &pixel = pixels[pixelIndex];
            //             if (!pixel.vp.beta.IsBlack()) {
            //                 // Add pixel's visible point to applicable grid cells
            //                 Float radius = pixel.radius;
            //                 Point3i pMin, pMax;
            //                 ToGrid(pixel.vp.p - Vector3f(radius, radius, radius),
            //                        gridBounds, gridRes, &pMin);
            //                 ToGrid(pixel.vp.p + Vector3f(radius, radius, radius),
            //                        gridBounds, gridRes, &pMax);
            //                 for (int z = pMin.z; z <= pMax.z; ++z)
            //                     for (int y = pMin.y; y <= pMax.y; ++y)
            //                         for (int x = pMin.x; x <= pMax.x; ++x) {
            //                             // Add visible point to grid cell $(x, y, z)$
            //                             int h = hash(Point3i(x, y, z), hashSize);
            //                             SPPMPixelListNode *node =
            //                                 arena.Alloc<SPPMPixelListNode>();
            //                             node->pixel = &pixel;

            //                             // Atomically add _node_ to the start of
            //                             // _grid[h]_'s linked list
            //                             node->next = grid[h];
            //                             while (grid[h].compare_exchange_weak(
            //                                        node->next, node) == false)
            //                                 ;
            //                         }
            //                 ReportValue(gridCellsPerVisiblePoint,
            //                             (1 + pMax.x - pMin.x) * (1 + pMax.y - pMin.y) *
            //                                 (1 + pMax.z - pMin.z));
            //             }
            //         }, nPixels, 4096);
            //     }

            //     // Trace photons and accumulate contributions
            //     {
            //         ProfilePhase _(Prof::SPPMPhotonPass);
            //         std::vector<MemoryArena> photonShootArenas(MaxThreadIndex());
            //         ParallelFor([&](int photonIndex) {
            //             MemoryArena &arena = photonShootArenas[ThreadIndex];
            //             // Follow photon path for _photonIndex_
            //             uint64_t haltonIndex =
            //                 (uint64_t)iter * (uint64_t)photonsPerIteration +
            //                 photonIndex;
            //             int haltonDim = 0;

            //             // Choose light to shoot photon from
            //             Float lightPdf;
            //             Float lightSample = RadicalInverse(haltonDim++, haltonIndex);
            //             int lightNum =
            //                 lightDistr->SampleDiscrete(lightSample, &lightPdf);
            //             const std::shared_ptr<Light> &light = scene.lights[lightNum];

            //             // Compute sample values for photon ray leaving light source
            //             Point2f uLight0(RadicalInverse(haltonDim, haltonIndex),
            //                             RadicalInverse(haltonDim + 1, haltonIndex));
            //             Point2f uLight1(RadicalInverse(haltonDim + 2, haltonIndex),
            //                             RadicalInverse(haltonDim + 3, haltonIndex));
            //             Float uLightTime =
            //                 Lerp(RadicalInverse(haltonDim + 4, haltonIndex),
            //                      camera->shutterOpen, camera->shutterClose);
            //             haltonDim += 5;

            //             // Generate _photonRay_ from light source and initialize _beta_
            //             RayDifferential photonRay;
            //             Normal3f nLight;
            //             Float pdfPos, pdfDir;
            //             Spectrum Le =
            //                 light->Sample_Le(uLight0, uLight1, uLightTime, &photonRay,
            //                                  &nLight, &pdfPos, &pdfDir);
            //             if (pdfPos == 0 || pdfDir == 0 || Le.IsBlack()) return;
            //             Spectrum beta = (AbsDot(nLight, photonRay.d) * Le) /
            //                             (lightPdf * pdfPos * pdfDir);
            //             if (beta.IsBlack()) return;

            //             // Follow photon path through scene and record intersections
            //             SurfaceInteraction isect;
            //             for (int depth = 0; depth < maxDepth; ++depth) {
            //                 if (!scene.Intersect(photonRay, &isect)) break;
            //                 ++totalPhotonSurfaceInteractions;
            //                 if (depth > 0) {
            //                     // Add photon contribution to nearby visible points
            //                     Point3i photonGridIndex;
            //                     if (ToGrid(isect.p, gridBounds, gridRes,
            //                                &photonGridIndex)) {
            //                         int h = hash(photonGridIndex, hashSize);
            //                         // Add photon contribution to visible points in
            //                         // _grid[h]_
            //                         for (SPPMPixelListNode *node =
            //                                  grid[h].load(std::memory_order_relaxed);
            //                              node != nullptr; node = node->next) {
            //                             ++visiblePointsChecked;
            //                             SPPMPixel &pixel = *node->pixel;
            //                             Float radius = pixel.radius;
            //                             if (DistanceSquared(pixel.vp.p, isect.p) >
            //                                 radius * radius)
            //                                 continue;
            //                             // Update _pixel_ $\Phi$ and $M$ for nearby
            //                             // photon
            //                             Vector3f wi = -photonRay.d;
            //                             Spectrum Phi =
            //                                 beta * pixel.vp.bsdf->f(pixel.vp.wo, wi);
            //                             for (int i = 0; i < Spectrum::nSamples; ++i)
            //                                 pixel.Phi[i].Add(Phi[i]);
            //                             ++pixel.M;
            //                         }
            //                     }
            //                 }
            //                 // Sample new photon ray direction

            //                 // Compute BSDF at photon intersection point
            //                 isect.ComputeScatteringFunctions(photonRay, arena, true,
            //                                                  TransportMode::Importance);
            //                 if (!isect.bsdf) {
            //                     --depth;
            //                     photonRay = isect.SpawnRay(photonRay.d);
            //                     continue;
            //                 }
            //                 const BSDF &photonBSDF = *isect.bsdf;

            //                 // Sample BSDF _fr_ and direction _wi_ for reflected photon
            //                 Vector3f wi, wo = -photonRay.d;
            //                 Float pdf;
            //                 BxDFType flags;

            //                 // Generate _bsdfSample_ for outgoing photon sample
            //                 Point2f bsdfSample(
            //                     RadicalInverse(haltonDim, haltonIndex),
            //                     RadicalInverse(haltonDim + 1, haltonIndex));
            //                 haltonDim += 2;
            //                 Spectrum fr = photonBSDF.Sample_f(wo, &wi, bsdfSample, &pdf,
            //                                                   BSDF_ALL, &flags);
            //                 if (fr.IsBlack() || pdf == 0.f) break;
            //                 Spectrum bnew =
            //                     beta * fr * AbsDot(wi, isect.shading.n) / pdf;

            //                 // Possibly terminate photon path with Russian roulette
            //                 Float q = std::max((Float)0, 1 - bnew.y() / beta.y());
            //                 if (RadicalInverse(haltonDim++, haltonIndex) < q) break;
            //                 beta = bnew / (1 - q);
            //                 photonRay = (RayDifferential)isect.SpawnRay(wi);
            //             }
            //             arena.Reset();
            //         }, photonsPerIteration, 8192);
            //         progress.Update();
            //         photonPaths += photonsPerIteration;
            //     }

            //     // Update pixel values from this pass's photons
            //     {
            //         ProfilePhase _(Prof::SPPMStatsUpdate);
            //         ParallelFor([&](int i) {
            //             SPPMPixel &p = pixels[i];
            //             if (p.M > 0) {
            //                 // Update pixel photon count, search radius, and $\tau$ from
            //                 // photons
            //                 Float gamma = (Float)2 / (Float)3;
            //                 Float Nnew = p.N + gamma * p.M;
            //                 Float Rnew = p.radius * std::sqrt(Nnew / (p.N + p.M));
            //                 Spectrum Phi;
            //                 for (int j = 0; j < Spectrum::nSamples; ++j)
            //                     Phi[j] = p.Phi[j];
            //                 p.tau = (p.tau + p.vp.beta * Phi) * (Rnew * Rnew) /
            //                         (p.radius * p.radius);
            //                 p.N = Nnew;
            //                 p.radius = Rnew;
            //                 p.M = 0;
            //                 for (int j = 0; j < Spectrum::nSamples; ++j)
            //                     p.Phi[j] = (Float)0;
            //             }
            //             // Reset _VisiblePoint_ in pixel
            //             p.vp.beta = 0.;
            //             p.vp.bsdf = nullptr;
            //         }, nPixels, 4096);
            //     }

            //     // Periodically store SPPM image in film and write image
            //     if (iter + 1 == nIterations || ((iter + 1) % writeFrequency) == 0) {
            //         int x0 = pixelBounds.pMin.x;
            //         int x1 = pixelBounds.pMax.x;
            //         uint64_t Np = (uint64_t)(iter + 1) * (uint64_t)photonsPerIteration;
            //         std::unique_ptr<Spectrum[]> image(new Spectrum[pixelBounds.Area()]);
            //         int offset = 0;
            //         for (int y = pixelBounds.pMin.y; y < pixelBounds.pMax.y; ++y) {
            //             for (int x = x0; x < x1; ++x) {
            //                 // Compute radiance _L_ for SPPM pixel _pixel_
            //                 const SPPMPixel &pixel =
            //                     pixels[(y - pixelBounds.pMin.y) * (x1 - x0) + (x - x0)];
            //                 Spectrum L = pixel.Ld / (iter + 1);
            //                 L += pixel.tau / (Np * Pi * pixel.radius * pixel.radius);
            //                 image[offset++] = L;
            //             }
            //         }
            //         camera->film->SetImage(image.get());
            //         camera->film->WriteImage();
            //         // Write SPPM radius image, if requested
            //         if (getenv("SPPM_RADIUS")) {
            //             std::unique_ptr<Float[]> rimg(
            //                 new Float[3 * pixelBounds.Area()]);
            //             Float minrad = 1e30f, maxrad = 0;
            //             for (int y = pixelBounds.pMin.y; y < pixelBounds.pMax.y; ++y) {
            //                 for (int x = x0; x < x1; ++x) {
            //                     const SPPMPixel &p =
            //                         pixels[(y - pixelBounds.pMin.y) * (x1 - x0) +
            //                                (x - x0)];
            //                     minrad = std::min(minrad, p.radius);
            //                     maxrad = std::max(maxrad, p.radius);
            //                 }
            //             }
            //             fprintf(stderr,
            //                     "iterations: %d (%.2f s) radius range: %f - %f\n",
            //                     iter + 1, progress.ElapsedMS() / 1000., minrad, maxrad);
            //             int offset = 0;
            //             for (int y = pixelBounds.pMin.y; y < pixelBounds.pMax.y; ++y) {
            //                 for (int x = x0; x < x1; ++x) {
            //                     const SPPMPixel &p =
            //                         pixels[(y - pixelBounds.pMin.y) * (x1 - x0) +
            //                                (x - x0)];
            //                     Float v = 1.f - (p.radius - minrad) / (maxrad - minrad);
            //                     rimg[offset++] = v;
            //                     rimg[offset++] = v;
            //                     rimg[offset++] = v;
            //                 }
            //             }
            //             Point2i res(pixelBounds.pMax.x - pixelBounds.pMin.x,
            //                         pixelBounds.pMax.y - pixelBounds.pMin.y);
            //             WriteImage("sppm_radius.png", rimg.get(), pixelBounds, res);
            //         }
            //     }
        }
    }
}
