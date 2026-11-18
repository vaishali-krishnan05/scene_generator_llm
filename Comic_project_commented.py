"""Commented companion to the Comic_project.ipynb code shared by the user.

The implementation is kept intentionally close to the supplied notebook.
Explanatory comments describe the existing behavior rather than changing the
generation algorithm.
"""

# PyTorch provides the model runtime, accelerator detection, and tensor
# precision settings used by the Diffusers pipeline.
import torch
# math and json support grid dimensions and reading the story's structured
# input file. os is imported in the supplied code but is not used below.
import os
import math
import json
import warnings

# These filters hide two common categories of dependency warnings. They do
# not hide exceptions, errors, or all possible warning categories.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Diffusers loads and runs the text-to-image pipeline. The Transformers image
# encoder and IP-Adapter connect a reference image to image generation.
from diffusers import AutoPipelineForText2Image
from transformers import CLIPVisionModelWithProjection

# Pillow objects are used to crop character references, compose the dialogue
# box, draw panel numbers, and build the final comic sheet. ImageOps is
# imported in the supplied code but is not used below.
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Compel converts long text prompts into embeddings for SDXL's two encoders.
from compel import Compel, ReturnedEmbeddingsType


# =========================================================
# 1. THE ENGINE WITH COMPEL & IP-ADAPTER
# =========================================================
class ComicGenerator:
    """Own model setup and generate one illustrated panel at a time.

    The instance retains character descriptions and first-seen reference
    images, so all panels in one story can share the same loaded model and
    attempt to keep recurring characters visually consistent.
    """

    def __init__(self, config_data, model_id="stabilityai/sdxl-turbo"):
        """Load the selected device, SDXL pipeline, Compel, and IP-Adapter.

        ``config_data`` is expected to contain ``config`` and ``characters``.
        The config supplies global style and negative prompt strings; each
        character entry supplies a text description.
        """
        # Prefer Apple's Metal Performance Shaders (MPS), then NVIDIA CUDA,
        # and finally CPU. This lets the same notebook run on several machine
        # types. Half precision is used on non-CPU devices; CPU uses float32.
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32

        # Read shared art-direction settings. These are appended to or applied
        # to every panel so the story has a consistent look.
        self.global_style = config_data['config']['global_style']
        self.negative_prompt = config_data['config']['negative_prompt']

        # Build runtime state for every character. The text description is
        # used in positive prompts; image_obj is filled with the character's
        # first panel to use later as an IP-Adapter reference.
        self.characters = {name: {"desc": d, "image_obj": None} for name, d in config_data['characters'].items()}

        print(f"🔄 Loading SDXL Pipeline...")

        # IP-Adapter uses a CLIP vision encoder to interpret reference images.
        # The encoder is loaded from the image_encoder subfolder of the
        # adapter repository and moved to the selected compute device.
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", dtype=self.dtype
        ).to(self.device)

        # Load the text-to-image model and supply the image encoder required by
        # IP-Adapter. The model_id argument allows the base diffusion model to
        # be changed when constructing ComicGenerator. The fp16 variant is
        # requested on accelerators, while CPU uses the default variant.
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, image_encoder=image_encoder, dtype=self.dtype, variant="fp16" if self.device != "cpu" else None
        ).to(self.device)

        # Configure Compel for both SDXL text encoders. It handles long
        # prompts and creates embeddings in the form expected by the
        # pipeline: token-level hidden states plus a pooled embedding.
        print(f"🚀 Initializing Compel for long-prompt support...")
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

        # Attach adapter weights to the text-to-image pipeline. At generation
        # time set_ip_adapter_scale controls how much the supplied image can
        # influence the output.
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl_vit-h.safetensors"
        )
        self.pipe.to(self.device)

    def crop_to_face(self, image):
        """Take a fixed central/upper crop for use as a character reference.

        This is a geometric crop, not a face detector: it assumes the
        character is roughly centered and near the upper part of the panel.
        """
        w, h = image.size
        return image.crop((w * 0.25, h * 0.1, w * 0.75, h * 0.6))

    def generate_panel(self, scene_desc, focus_chars):
        """Create a single panel image using its scene and character focus.

        The scene and character descriptions control content through text
        embeddings. A reference image is additionally used for the first
        focused character if that character has already appeared in an earlier
        panel.
        """
        # Collect the text descriptions for known focused characters. Unknown
        # names are skipped in this prompt-building step; note that the later
        # IP-Adapter lookup accesses the first focus name directly, so an
        # unknown first name can still raise a KeyError there.
        char_prompts = [self.characters[name]['desc'] for name in focus_chars if name in self.characters]
        # Construct one positive prompt from the current action, the
        # characters' stable visual descriptions, and the shared art style.
        full_prompt = f"{scene_desc}, {', '.join(char_prompts)}, {self.global_style}"

        # 1. PROCESS LONG PROMPT WITH COMPEL
        # Compel runs the positive and negative strings through both SDXL text
        # encoders and returns the token-level and pooled embeddings needed
        # for the pipeline call below.
        conditioning, pooled = self.compel(full_prompt)
        neg_conditioning, neg_pooled = self.compel(self.negative_prompt)

        # 2. IP-ADAPTER LOGIC
        # Use the first focused character as the single reference-image
        # source. Until this character has a stored image, set the adapter
        # scale to zero so the blank placeholder does not guide generation.
        primary_char = focus_chars[0] if focus_chars else None
        ref_img = None
        scale = 0.0

        if primary_char and self.characters[primary_char]['image_obj']:
            # Crop the first panel containing this character to emphasize its
            # face/upper body and reduce unrelated scene content.
            ref_img = self.crop_to_face(self.characters[primary_char]['image_obj'])
            scale = 0.7

        # Set the adapter's strength for this generation. The pipeline
        # receives an image argument in either case; when no reference exists,
        # use a black square placeholder with adapter influence disabled.
        self.pipe.set_ip_adapter_scale(scale)
        placeholder = ref_img if ref_img else Image.new("RGB", (1024, 1024), (0,0,0))

        # 3. GENERATE USING EMBEDDS INSTEAD OF PROMPT STRING
        # Supplying embeddings avoids asking the pipeline to tokenize the
        # prompt again. The call combines positive and negative conditioning,
        # the IP-Adapter image, and a short SDXL Turbo inference schedule.
        print(f"🎨 Generating: {scene_desc[:50]}...")
        image = self.pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_conditioning,
            negative_pooled_prompt_embeds=neg_pooled,
            ip_adapter_image=placeholder,
            num_inference_steps=4,
            guidance_scale=0.0
        ).images[0]

        # Record the generated panel as the initial reference for each known
        # focused character that has no reference yet. Later appearances keep
        # using this first reference rather than replacing it.
        for name in focus_chars:
            if name in self.characters and self.characters[name]['image_obj'] is None:
                self.characters[name]['image_obj'] = image

        return image


# =========================================================
# 2. THE REST OF THE UTILS (TEXT & RUNNER)
# =========================================================
def add_text_box(image, text, panel_num):
    """Overlay a wrapped dialogue box and panel-number badge on an image."""
    draw = ImageDraw.Draw(image)
    W, H = image.size

    # Prefer a bold Arial font installed on the system. If it is unavailable,
    # Pillow's default font is used so the program can still render captions.
    try: font = ImageFont.truetype("Arial Bold.ttf", size=24)
    except: font = ImageFont.load_default()

    # Split the dialogue into words and accumulate text while its measured
    # width fits within the panel, leaving 80 pixels for horizontal margins.
    # If even the first word is wider than that limit, this implementation
    # still appends the empty current line before starting a new one.
    words = text.split(); lines = []; curr = ""
    for w in words:
        if font.getlength(curr + w) < W - 80: curr += w + " "
        else: lines.append(curr); curr = w + " "
    lines.append(curr)
    wrapped = "\n".join(lines)

    # Use Pillow's measured multiline text bounds to size the caption height,
    # with 40 pixels of additional padding.
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    h = (bbox[3]-bbox[1]) + 40

    # Draw the caption rectangle on a separate transparent layer, then
    # composite that layer over the illustration. The 240 alpha value makes
    # the white background nearly opaque while preserving a little image
    # visibility beneath it.
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([10, H-h-10, W-10, H-10], fill=(255,255,255,240), outline=(0,0,0), width=3)
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    # Draw the caption centered in the lower box. The "mm" anchor means the
    # given coordinate is the middle of the text block.
    ImageDraw.Draw(image).multiline_text((W/2, H-h/2-10), wrapped, font=font, fill=(0,0,0), anchor="mm", align="center")

    # Draw a black square at the upper-left and place the 1-based panel
    # number inside it in white, so the intended reading order is visible.
    draw_num = ImageDraw.Draw(image)
    draw_num.rectangle([0, 0, 45, 45], fill=(0, 0, 0))
    draw_num.text((15, 8), str(panel_num), fill=(255, 255, 255), font=font)
    return image.convert("RGB")


def assemble_comic(panels, output_name):
    """Arrange panel images in a left-to-right, top-to-bottom two-column grid."""
    # Use the first panel's size as the cell size. This assumes every panel
    # has the same dimensions, which is true for images from one pipeline.
    w, h = panels[0].size
    cols = 2
    rows = math.ceil(len(panels) / cols)

    # Allocate a white background large enough for all rows. An odd final
    # panel count leaves the bottom-right cell empty.
    grid = Image.new('RGB', (w * cols, h * rows), (255, 255, 255))

    # Remainder gives each panel's column; integer division gives its row.
    for i, p in enumerate(panels):
        grid.paste(p, ((i % cols) * w, (i // cols) * h))

    grid.save(output_name)
    grid.show()
    print(f"✅ Comic saved as {output_name}")


def run_pipeline(json_path):
    """Load a story, create all captioned panels, and save the final comic."""
    # Read the JSON file into Python dictionaries/lists. Required keys are
    # accessed below; malformed or incomplete input raises an error.
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Keep one engine for the entire story so model loading happens only once
    # and its per-character reference-image state survives between panels.
    engine = ComicGenerator(data)
    panels = []

    # Walk story panels in their defined order. The displayed panel number is
    # one-based, while enumerate's default index starts at zero.
    for i, step in enumerate(data['script']):
        raw_img = engine.generate_panel(step['scene'], step['focus'])
        panel = add_text_box(raw_img, step['dialogue'], i + 1)
        panels.append(panel)
        # Preserve a separate PNG for each finished panel in the current
        # working directory.
        panel.save(f"panel_{i+1}.png")

    # Assemble final grid
    # This runner builds and saves the final grid directly (rather than
    # calling assemble_comic). The output filename is fixed for this story.
    w, h = panels[0].size
    grid = Image.new('RGB', (w*2, math.ceil(len(panels)/2)*h))
    for i, p in enumerate(panels):
        grid.paste(p, ((i%2)*w, (i//2)*h))
    grid.save("diwali_sweets.png")
    grid.show()


if __name__ == "__main__":
    # Running this module directly starts the generation process using the
    # expected story JSON in the current working directory. Importing the
    # file from another Python module will not run the pipeline.
    run_pipeline("diwali_sweets.json")
