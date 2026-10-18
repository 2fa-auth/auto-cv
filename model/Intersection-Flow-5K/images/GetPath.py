import os

def list_image_paths(image_dir, output_file):

    image_paths = [os.path.join(os.path.abspath(image_dir), img) for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    with open(output_file, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")

if __name__ == "__main__":
    image_dir = './train' 
    output_file = '../train.txt' 
    list_image_paths(image_dir, output_file)
    print(f"Image paths saved to {output_file}")
    image_dir = './val'  
    output_file = '../val.txt'  
    list_image_paths(image_dir, output_file)
    print(f"Image paths saved to {output_file}")

    image_dir = './test'  
    output_file = '../test.txt'
    list_image_paths(image_dir, output_file)
    print(f"Image paths saved to {output_file}")

