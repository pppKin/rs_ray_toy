use crate::{
    filters::{Filter, IFilter},
    geometry::{
        bnd2_intersect_bnd2, pnt2_ceil, pnt2_floor, pnt2_max_pnt2, pnt2_min_pnt2, Bounds2f,
        Bounds2i, Point2f, Point2i, Vector2f,
    },
    spectrum::{xyz_to_rgb, ISpectrum, Spectrum},
    {write_image, SPECTRUM_N},
};
use std::sync::{Arc, RwLock};

const FILTER_TABLE_WIDTH: usize = 16;
#[derive(Debug, Default, Copy, Clone)]
pub struct Pixel {
    // Float xyz[3];
    xyz: [f64; 3],
    filter_weight_sum: f64,
    splat_xyz: [f64; 3],
}

// FilmTilePixel Declarations
#[derive(Debug, Default, Clone)]
pub struct FilmTilePixel {
    contrib_sum: Spectrum<SPECTRUM_N>,
    filter_weight_sum: f64,
}

#[derive(Debug)]
pub struct Film<T>
where
    T: IFilter,
{
    // Film Public Data
    pub full_resolution: Point2i,
    pub diagonal: f64,
    pub filter: Box<Filter<T>>,
    pub filename: String,
    pub cropped_pixel_bounds: Bounds2i,

    // Film Private Data
    pixels: RwLock<Vec<Pixel>>,
    filter_table: [f64; FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH],
    scale: f64,
    max_sample_luminance: f64,
}

#[derive(Debug, Default, Clone)]
pub struct FilmTile<'a> {
    pixel_bounds: Bounds2i,
    filter_radius: Vector2f,
    inv_filter_radius: Vector2f,

    filter_table: &'a [f64],
    filter_table_size: usize,
    pixels: Vec<FilmTilePixel>,
    max_sample_luminance: f64,
}

impl<'a> FilmTile<'a> {
    pub fn new(
        pixel_bounds: Bounds2i,
        filter_radius: Vector2f,
        filter_table: &'a [f64],
        filter_table_size: usize,
        max_sample_luminance: f64,
    ) -> Self {
        let inv_filter_radius = Vector2f::new(1.0 / filter_radius.x, 1.0 / filter_radius.y);
        let pixels = Vec::<FilmTilePixel>::with_capacity(0_usize.max(pixel_bounds.area() as usize));
        Self {
            pixel_bounds,
            filter_radius,
            inv_filter_radius,
            filter_table,
            filter_table_size,
            pixels,
            max_sample_luminance,
        }
    }
    pub fn add_sample(
        &mut self,
        p_film: &Point2f,
        L: &mut Spectrum<SPECTRUM_N>,
        sample_weight: f64,
    ) {
        if L.y() > self.max_sample_luminance {
            *L *= self.max_sample_luminance / L.y();
        }
        // Compute sample's raster bounds
        let p_film_discrete = *p_film - Vector2f::new(0.5, 0.5);
        let mut p0 = Point2i::from(pnt2_ceil(p_film_discrete - self.filter_radius));
        let mut p1 = Point2i::from(p_film_discrete + self.filter_radius) + Point2i::new(1, 1);
        p0 = pnt2_max_pnt2(p0, self.pixel_bounds.p_min);
        p1 = pnt2_min_pnt2(p1, self.pixel_bounds.p_max);

        // Loop over filter support and add sample to pixel arrays
        // Precompute $x$ and $y$ filter table offsets
        let mut ifx = vec![0_i64; (p1.x - p1.x) as usize];
        for x in p0.x..p1.x {
            let fx = f64::abs(
                (x as f64 - p_film_discrete.x)
                    * self.inv_filter_radius.x
                    * self.filter_table_size as f64,
            );
            ifx[(x - p0.x) as usize] =
                i64::min(fx.floor() as i64, (self.filter_table_size - 1) as i64);
        }

        let mut ify = vec![0_i64; (p1.x - p1.x) as usize];
        for y in p0.y..p1.y {
            let fx = f64::abs(
                (y as f64 - p_film_discrete.y)
                    * self.inv_filter_radius.y
                    * self.filter_table_size as f64,
            );
            ify[(y - p0.y) as usize] =
                i64::min(fx.floor() as i64, (self.filter_table_size - 1) as i64);
        }

        for y in p0.y..p1.y {
            for x in p0.x..p1.x {
                // Evaluate filter value at $(x,y)$ pixel
                let offset = ify[(y - p0.y) as usize] * self.filter_table_size as i64
                    + ifx[(x - p0.x) as usize];
                let filter_weight = self.filter_table[offset as usize];

                // Update pixel values with filtered sample contribution
                let pixel = self.get_pixel(&Point2i::new(x, y));
                pixel[0].contrib_sum += (*L * sample_weight) * filter_weight;
                pixel[0].filter_weight_sum += filter_weight;
            }
        }
    }
    pub fn get_pixel(&mut self, p: &Point2i) -> &mut [FilmTilePixel] {
        assert!(Bounds2i::inside_exclusive(p, &self.pixel_bounds));
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        let offset = (p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width;
        &mut self.pixels[offset as usize..]
    }
    pub fn get_pixel_bounds(&self) -> Bounds2i {
        self.pixel_bounds
    }
}

impl<T> Film<T>
where
    T: IFilter,
{
    pub fn new(
        full_resolution: Point2i,
        diagonal: f64,
        mut filter: Box<Filter<T>>,
        filename: String,
        cropped_window: Bounds2f,
        scale: f64,
        max_sample_luminance: f64,
    ) -> Self {
        let cropped_pixel_bounds = Bounds2i::new(
            Point2i::new(
                f64::ceil(full_resolution.x as f64 * cropped_window.p_min.x) as i64,
                f64::ceil(full_resolution.y as f64 * cropped_window.p_min.y) as i64,
            ),
            Point2i::new(
                f64::ceil(full_resolution.x as f64 * cropped_window.p_max.x) as i64,
                f64::ceil(full_resolution.y as f64 * cropped_window.p_max.y) as i64,
            ),
        );
        let pixels = RwLock::new(Vec::<Pixel>::with_capacity(
            cropped_pixel_bounds.area() as usize
        ));
        // Precompute filter weight table
        let mut offset = 0;
        let mut filter_table = [0_f64; FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH];
        for y in 0..FILTER_TABLE_WIDTH {
            for x in 0..FILTER_TABLE_WIDTH {
                let mut p = Point2f::default();
                p.x = (x as f64 + 0.5) * filter.r.radius.x / (FILTER_TABLE_WIDTH as f64);
                p.x = (y as f64 + 0.5) * filter.r.radius.y / (FILTER_TABLE_WIDTH as f64);
                filter_table[offset] = filter.evaluate(&p);
                offset += 1;
            }
        }

        Self {
            full_resolution,
            diagonal,
            filter,
            filename,
            cropped_pixel_bounds,
            pixels,
            filter_table,
            scale,
            max_sample_luminance,
        }
    }
    pub fn get_sample_bounds(&self) -> Bounds2i {
        let p1 = pnt2_floor(
            Point2f::new(
                self.cropped_pixel_bounds.p_min.x as f64,
                self.cropped_pixel_bounds.p_min.y as f64,
            ) + Vector2f::new(0.5, 0.5)
                - self.filter.r.radius,
        );
        let p2 = pnt2_ceil(
            Point2f::new(
                self.cropped_pixel_bounds.p_max.x as f64,
                self.cropped_pixel_bounds.p_max.y as f64,
            ) - Vector2f::new(0.5, 0.5)
                + self.filter.r.radius,
        );

        Bounds2i::new(
            Point2i::new(p1.x as i64, p2.y as i64),
            Point2i::new(p2.x as i64, p2.y as i64),
        )
    }
    pub fn get_physical_extent(&self) -> Bounds2f {
        let aspect = self.full_resolution.y as f64 / self.full_resolution.x as f64;
        let x = (self.diagonal * self.diagonal / (1_f64 + aspect * aspect)).sqrt();
        let y = aspect * x;
        Bounds2f::new(
            Point2f::new(-x / 2.0, -y / 2.0),
            Point2f::new(x / 2.0, y / 2.0),
        )
    }
    pub fn get_pixel_offset(&self, p: &Point2i) -> usize {
        assert!(Bounds2i::inside_exclusive(p, &self.cropped_pixel_bounds));
        let width = self.cropped_pixel_bounds.p_max.x - self.cropped_pixel_bounds.p_min.x;
        let offset = (p.x - self.cropped_pixel_bounds.p_min.x)
            + (p.y - self.cropped_pixel_bounds.p_min.y) * width;
        offset as usize
    }
    pub fn get_film_tile(&self, sample_bounds: &Bounds2i) -> FilmTile
where {
        // Bound image pixels that samples in _sampleBounds_ contribute to
        let half_pixel = Vector2f::new(0.5, 0.5);
        let f_bounds = Bounds2f::from(*sample_bounds);

        let p0: Point2i = pnt2_ceil(f_bounds.p_min - half_pixel - self.filter.r.radius).into();
        let p1: Point2i = Point2i::from(pnt2_floor(
            f_bounds.p_max - half_pixel + self.filter.r.radius,
        )) + Point2i::new(1, 1);
        let tile_pixel_bounds =
            bnd2_intersect_bnd2(&Bounds2i::new(p0, p1), &self.cropped_pixel_bounds);
        FilmTile::new(
            tile_pixel_bounds,
            self.filter.r.radius,
            &self.filter_table,
            FILTER_TABLE_WIDTH,
            self.max_sample_luminance,
        )
    }

    pub fn clear(&mut self) {
        for p in self.cropped_pixel_bounds.into_iter() {
            for c in 0..3 {
                let mut pixel =
                    self.pixels.read().expect("Error Getting Pixel")[self.get_pixel_offset(&p)];
                pixel.splat_xyz[c] = 0_f64;
                pixel.xyz[c] = 0_f64;
                pixel.filter_weight_sum = 0_f64;
            }
        }
    }

    pub fn MergeFilmTile(&self, tile: Arc<RwLock<FilmTile>>) {
        let mut t = tile.write().expect("Error preparing tile for writing");
        for p in t.get_pixel_bounds().into_iter() {
            // Merge _pixel_ into _Film::pixels_
            let tile_pixels = t.get_pixel(&p);
            let merge_pixels_offset = self.get_pixel_offset(&p);
            let xyz = tile_pixels[0].contrib_sum.to_xyz();
            let mut pixels = self
                .pixels
                .write()
                .expect("Error preparing pixels for writing");
            for i in 0..3 {
                pixels[merge_pixels_offset].xyz[i] += xyz[i];
                pixels[merge_pixels_offset].filter_weight_sum += tile_pixels[0].filter_weight_sum;
            }
        }
    }

    pub fn set_image(&self, img: &[Spectrum<SPECTRUM_N>]) {
        let n_pixels = self.cropped_pixel_bounds.area() as usize;
        let mut pixels = self
            .pixels
            .write()
            .expect("Error preparing pixels for writing");
        for i in 0..n_pixels {
            pixels[i].xyz = img[i].to_xyz();
            pixels[i].filter_weight_sum = 1_f64;
            pixels[i].splat_xyz[0] = 0_f64;
            pixels[i].splat_xyz[1] = 0_f64;
            pixels[i].splat_xyz[2] = 0_f64;
        }
    }

    pub fn add_splat(&self, p: &Point2f, v: &mut Spectrum<SPECTRUM_N>) {
        if v.has_nan() {
            eprintln!(
                "Ignoring splatted spectrum with NaN values at ({}, {})",
                p.x, p.y
            );
            return;
        } else if v.y() < 0.0 {
            eprintln!(
                "Ignoring splatted spectrum with negative luminance {} at ({}, {})",
                v.y(),
                p.x,
                p.y
            );
            return;
        } else if v.y().is_infinite() {
            eprintln!(
                "Ignoring splatted spectrum with infinite luminance at ({}, {})",
                p.x, p.y
            );
            return;
        }

        let pi = Point2i::from(pnt2_floor(*p));
        if !Bounds2i::inside_exclusive(&pi, &self.cropped_pixel_bounds) {
            return;
        }

        if v.y() > self.max_sample_luminance {
            *v *= self.max_sample_luminance / v.y();
        }
        let xyz = v.to_xyz();

        let pixel_offset = self.get_pixel_offset(&pi);
        let mut pixels = self
            .pixels
            .write()
            .expect("Error preparing pixels for writing");
        for i in 0..3 {
            pixels[pixel_offset].splat_xyz[i] += xyz[i];
        }
    }

    pub fn write_image(&self, splat_scale: f64) {
        // Convert image to RGB and compute final pixel values
        println!("Converting image to RGB and computing final weighted pixel values");
        let mut rgb = vec![0_f64; (self.cropped_pixel_bounds.area() * 3) as usize];

        let mut offset = 0;
        let mut pixels = self
            .pixels
            .write()
            .expect("Error preparing pixels for writing");
        for p in self.cropped_pixel_bounds.into_iter() {
            // Convert pixel XYZ color to RGB
            let p_offset = self.get_pixel_offset(&p);
            let pixel = &mut pixels[p_offset];
            xyz_to_rgb(&mut pixel.xyz, &mut rgb[3 * offset..3 * offset + 3]);
            // Normalize pixel with weight sum
            let filter_weight_sum = pixel.filter_weight_sum;
            if filter_weight_sum != 0_f64 {
                let inv_wt = 1.0 / filter_weight_sum;
                rgb[3 * offset] = 0_f64.max(rgb[3 * offset] * inv_wt);
                rgb[3 * offset + 1] = 0_f64.max(rgb[3 * offset + 1] * inv_wt);
                rgb[3 * offset + 2] = 0_f64.max(rgb[3 * offset + 2] * inv_wt);
            }

            // Add splat value at pixel
            let mut splat_rgb = [0_f64; 3];
            let mut splat_xyz = pixel.splat_xyz;
            xyz_to_rgb(&mut splat_xyz, &mut splat_rgb);

            rgb[3 * offset] += splat_scale * splat_rgb[0];
            rgb[3 * offset + 1] += splat_scale * splat_rgb[1];
            rgb[3 * offset + 2] += splat_scale * splat_rgb[2];

            // Scale pixel value by _scale_
            rgb[3 * offset] *= self.scale;
            rgb[3 * offset + 1] *= self.scale;
            rgb[3 * offset + 2] *= self.scale;
            offset += 1;
        }

        // Write RGB image
        write_image(
            &self.filename,
            &rgb,
            self.cropped_pixel_bounds,
            self.full_resolution,
        )
        .expect("FAILED writing to image!");
    }
}
