# Preparing your data

In this tutorial, the data is already ready for you to extract the information you need and compare it. For Picanteo to work optimally, the data must be :
- geometrically coherent
- radiometrically corrected
- with panchromatic and multispectral data of the same height and width
- if only the 2D pipeline is used: projected on the ground (which is not the case in this tutorial)

The purpose of this "readme" is to show you what processing we have applied to prepare this data.

## make the data geometrically coherent

In photogrammetry, bundle adjustment is necessary operation in order to produce coherent (without planimetric or altimetric offsets) 3D models. The coherence of 3D models and images is very important for PICANTEO so that they can be compared correctly.

Classically, homologous points are detected in all the images. The lines of sight from each image for these homologous points must intersect at a single point. The geometric models associated with each image are modified so that this is the case.

The executable [cars-bundleadjustment](https://github.com/CNES/cars/blob/master/cars/bundleadjustment.py) available in CARS can be used to perform this operation.

## correct the radiometry of images

"Which pixels correspond to areas of water? to vegetated areas?" These are the questions we'll be asking ourselves in the pipeline, in particular to filter out areas of interest. To do this, the images must be radiometrically comparable.

Image pixels are generally not calibrated in physically significant units. We say that these pixels are expressed in **Digital Number** or **DN**.

**Reflectance** is the proportion of radiation striking a surface in relation to the radiation reflected by that surface: so the pixels in the images need to be converted to this unit in order to calculate radiometric indices correctly.

The application [OpticalCalibration](https://www.orfeo-toolbox.org/CookBook/Applications/app_OpticalCalibration.html
) from the Orfeo Toolbox allows converting pixel values from DN (for Digital Numbers) to reflectance.

## fuse the panchromatic image with the multispectral image

The principle of image acquisition, whether in satellite or more generally in photography, is to capture photons in a radiometric band: The wider the band, the more photos are acquired and the sharper the image. 

This is why for a Pleiades image, for example, we acquire a panchromatic band at 70cm resolution (spectral band corresponding to the visible range) and a multispectral image (red, green, blue, infrared) with 2m80 resolution.

To use PICANTEO's 3D pipeline, we recommend ordering the panchromatic and multispectral images separately (bundle) because :
- 3D reconstruction works best with panchromatic images: as this is the sharpest image available.
- to obtain a colour image of the same size as the panchromatic image, simply perform the pancharpening operation yourself.

The application [BundleToPerfectSensor](https://www.orfeo-toolbox.org/CookBook/Applications/app_BundleToPerfectSensor.html) performs P+XS pansharpening. 