from PIL import Image
import glob
import re
import collections

# Create the frames
frames = []
imgs = glob.glob(r"../figures/von_karman_vortex_shedding/all_png/*.png")
regex = re.compile(r'\d+')
numbers = [int(x) for img in imgs for x in regex.findall(img)]

img_dict = {
    img: number for img, number in zip(imgs, numbers)
}

ordered_img_dict = collections.OrderedDict(sorted(img_dict.items(), key=lambda item: item[1]))

for img, _ in ordered_img_dict.items():
    new_frame = Image.open(img)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('../figures/von_karman_vortex_shedding/png_to_gif.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=10, loop=0)
