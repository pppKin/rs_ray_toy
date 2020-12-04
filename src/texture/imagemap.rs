use std::{
    fmt::Debug,
    ops::{Add, Mul},
};

use crate::{geometry::Vector2f, interaction::SurfaceInteraction, mipmap::ImageWrap};

use super::{Texture, TextureMapping2D};

// TexInfo Declarations
#[derive(Debug, PartialEq, PartialOrd)]
pub struct TexInfo {
    filename: String,
    do_trilinear: bool,
    max_aniso: f64,

    wrap_mode: ImageWrap,
    scale: f64,
    gamma: bool,
}

// ImageTexture Declarations
// template <typename Tmemory, typename Treturn>
// class ImageTexture : public Texture<Treturn> {
//   public:
//     // ImageTexture Public Methods
//     ImageTexture(std::unique_ptr<TextureMapping2D> m,
//                  const std::string &filename, bool doTri, Float maxAniso,
//                  ImageWrap wm, Float scale, bool gamma);
//     static void ClearCache() {
//         textures.erase(textures.begin(), textures.end());
//     }
//     Treturn Evaluate(const SurfaceInteraction &si) const {
//         Vector2f dstdx, dstdy;
//         Point2f st = mapping->Map(si, &dstdx, &dstdy);
//         Tmemory mem = mipmap->Lookup(st, dstdx, dstdy);
//         Treturn ret;
//         convertOut(mem, &ret);
//         return ret;
//     }

//   private:
//     // ImageTexture Private Methods
//     static MIPMap<Tmemory> *GetTexture(const std::string &filename,
//                                        bool doTrilinear, Float maxAniso,
//                                        ImageWrap wm, Float scale, bool gamma);
//     static void convertIn(const RGBSpectrum &from, RGBSpectrum *to, Float scale,
//                           bool gamma) {
//         for (int i = 0; i < RGBSpectrum::nSamples; ++i)
//             (*to)[i] = scale * (gamma ? InverseGammaCorrect(from[i]) : from[i]);
//     }
//     static void convertIn(const RGBSpectrum &from, Float *to, Float scale,
//                           bool gamma) {
//         *to = scale * (gamma ? InverseGammaCorrect(from.y()) : from.y());
//     }
//     static void convertOut(const RGBSpectrum &from, Spectrum *to) {
//         Float rgb[3];
//         from.ToRGB(rgb);
//         *to = Spectrum::FromRGB(rgb);
//     }
//     static void convertOut(Float from, Float *to) { *to = from; }

//     // ImageTexture Private Data
//     std::unique_ptr<TextureMapping2D> mapping;
//     MIPMap<Tmemory> *mipmap;
//     static std::map<TexInfo, std::unique_ptr<MIPMap<Tmemory>>> textures;
// };

// #[derive(Debug)]
// pub struct ImageTexture<T, U>{
//     mapping: Box<dyn TextureMapping2D>,
// }
