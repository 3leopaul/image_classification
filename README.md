In this notebook, I use the CLIP model to perform image–text classification.
I load sample images and a set of text prompts (for example, "a photo of a cat" and "a photo of a dog"), process them
with the CLIP processor, and then use the CLIP model to calculate similarity scores between each image and each text description.
Finally, I identify the most probable class for each image and display its confidence score.

Libraries used:
- Pillow (PIL): to load and handle image files.
- Torch (PyTorch): to manage tensors and run the model computations.
- Transformers (from Hugging Face): to load the CLIP model and its associated processor.
