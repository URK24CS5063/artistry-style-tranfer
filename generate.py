from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# -------- LOAD RANDOM IMAGE --------
def get_random_image(folder):
    files = os.listdir(folder)
    img_name = random.choice(files)
    return os.path.join(folder, img_name)

content_path = get_random_image("dataset/trainA")
style_path = get_random_image("dataset/trainB")

content = np.array(Image.open(content_path).resize((256,256))) / 255.0
style = np.array(Image.open(style_path).resize((256,256))) / 255.0

# -------- LOSS FUNCTIONS --------
def content_loss(output, content):
    return np.mean((output - content) ** 2)

def style_loss(output, style):
    return np.mean((output - style) ** 2)

# -------- RL OPTIMIZATION --------
alphas = np.linspace(0.1, 1.0, 10)
loss_values = []

best_output = None
best_loss = float('inf')

for alpha in alphas:

    output = (1 - alpha) * content + alpha * style

    c_loss = content_loss(output, content)
    s_loss = style_loss(output, style)

    total_loss = c_loss + 0.5 * s_loss

    loss_values.append(total_loss)

    if total_loss < best_loss:
        best_loss = total_loss
        best_output = output

# -------- SAVE IMAGE --------
best_output = (best_output * 255).astype(np.uint8)
Image.fromarray(best_output).save("result.png")

# -------- SAVE LOSS GRAPH --------
plt.figure()
plt.plot(alphas, loss_values)
plt.xlabel("Alpha (Style Strength)")
plt.ylabel("Loss")
plt.title("RL Optimization Loss Graph")
plt.savefig("loss.png")
plt.close()

print("✅ Output + Loss graph generated")