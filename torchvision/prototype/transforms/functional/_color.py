from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT

adjust_brightness_image_tensor = _FT.adjust_brightness
adjust_brightness_image_pil = _FP.adjust_brightness

adjust_saturation_image_tensor = _FT.adjust_saturation
adjust_saturation_image_pil = _FP.adjust_saturation

adjust_contrast_image_tensor = _FT.adjust_contrast
adjust_contrast_image_pil = _FP.adjust_contrast

adjust_sharpness_image_tensor = _FT.adjust_sharpness
adjust_sharpness_image_pil = _FP.adjust_sharpness

posterize_image_tensor = _FT.posterize
posterize_image_pil = _FP.posterize

solarize_image_tensor = _FT.solarize
solarize_image_pil = _FP.solarize

autocontrast_image_tensor = _FT.autocontrast
autocontrast_image_pil = _FP.autocontrast

equalize_image_tensor = _FT.equalize
equalize_image_pil = _FP.equalize

invert_image_tensor = _FT.invert
invert_image_pil = _FP.invert

adjust_hue_image_tensor = _FT.adjust_hue
adjust_hue_image_pil = _FP.adjust_hue

adjust_gamma_image_tensor = _FT.adjust_gamma
adjust_gamma_image_pil = _FP.adjust_gamma
