# Image Roots

Original image folders are not stored in Git. This directory keeps only
augmented image placeholders and documentation.

Expected local links after running `scripts/setup_local_assets.sh`:

- `echosight_images`
- `echosight_inat_val_id2name.json`
- `evqa_landmark_images`
- `evqa_val_images`
- `infoseek_val_images`
- `infoseek_wikipedia_images_full`
- `reflectiva_evqa_inference_images`
- `reflectiva_infoseek_val_image`

Augmented composite-image placeholders remain under `data/images/augmented/`.
The augmented CSVs use relative image paths matching the HuggingFace dataset
repo `VanQianMa/ECCV26-ARA`.
