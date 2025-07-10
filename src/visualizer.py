"""
Interactive image viewer to visualize predictions made by the network on the test set.

Supports:
- Viewing single images one-by-one
- Viewing batches of images at once (user-defined batch size)
- User-defined image range and view mode
- Pauses for user to continue
"""

import matplotlib.pyplot as plt
import numpy as np

def preview_predictions(net, input_test):
    total_images = len(input_test)

    # Get starting index
    while True:
        try:
            start = int(input(f"Enter the starting index (0 to {total_images - 1}): "))
            if start < 0 or start >= total_images:
                print(f"Invalid start index. Must be between 0 and {total_images - 1}.")
            else:
                break
        except ValueError:
            print("Please enter a valid integer.")

    # Get how many images to show
    while True:
        try:
            count = int(input(f"How many images do you want to preview (1 to {total_images - start})? "))
            if count <= 0 or start + count > total_images:
                print(f"Invalid count. Must be between 1 and {total_images - start}.")
            else:
                break
        except ValueError:
            print("Please enter a valid integer.")

    end = start + count

    # Choose display mode
    while True:
        mode = input("View mode — type 'one' for one-by-one, or 'batch' for a groups with variable number of images ").strip().lower()
        if mode in ['one', 'batch']:
            break
        else:
            print("Invalid mode. Please type 'one' or 'batch'.")

    # One-by-one image viewer
    if mode == "one":
        for i in range(start, end):
            image = input_test[i].reshape(28, 28)
            output = net.forward(input_test[i].reshape(1, 784))
            predicted_label = np.argmax(output)

            fig = plt.figure()
            fig.canvas.manager.set_window_title(f"Image {i + 1}")
            plt.imshow(image, cmap='gray')
            plt.title(f"Image {i + 1} — Predicted Label: {predicted_label}")
            plt.axis('off')
            plt.show(block=False)

            input("Press Enter to continue to the next image.")
            plt.close()

    # Batch image viewer 
    elif mode == "batch":
        while True:
            try:
                batch_size = int(input("Enter how many images to display per batch: "))
                if batch_size <= 0 or batch_size > count:
                    print(f"Invalid batch size. Must be between 1 and {count}.")
                else:
                    break
            except ValueError:
                print("Please enter a valid integer.")

        for i in range(start, end, batch_size):
            batch_end = min(i + batch_size, end)
            batch = input_test[i:batch_end]

            plt.figure(figsize=(batch_size * 1.5, 4))
            for j, img_flat in enumerate(batch):
                img = img_flat.reshape(28, 28)
                output = net.forward(img_flat.reshape(1, 784))
                predicted = np.argmax(output)

                plt.subplot(1, len(batch), j + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f"#{i + j + 1}\nPred: {predicted}")
                plt.axis('off')

            plt.suptitle(f"Images {i + 1} to {batch_end}")
            plt.show()

            if batch_end < end:
                input("Press Enter to view the next batch of images.")