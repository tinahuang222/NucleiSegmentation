# NucleiSegmentation
Instance nuclei segmentation


## Nuclear Mask Generation Process

1. Initialize python environment to correspond to packages in `nuclear_mask_generation/requirements.txt`
2. Modify the example configuration file at `nuclear_mask_generation/generate_masks_config.json` to match local input/output directories
3. Enter directory for mask generation `cd nuclear_mask_generation`
4. Run with command `python generate_nuclear_masks.py generate_masks_config.json `


__Approximate Runtime:__ On 1000 images takes about 8 hours on CPU
