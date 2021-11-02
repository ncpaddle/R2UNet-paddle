import numpy as np

def gen_fake_data():
    fake_data = np.random.rand(1, 3, 48, 48).astype(np.float32)
    np.save("fake_data.npy", fake_data)


def gen_fake_label():
    fake_label = np.random.rand(1, 1, 48, 48).astype(np.float32)
    print(fake_label)
    np.save("fake_label.npy", fake_label)


def gen_fake_data2():
   fake_img =  (np.random.rand(560, 560, 3) * 255.).astype(np.uint8)
   fake_target = np.random.randint(low=0, high=2, size=(560, 560)) * 255
   fake_mask = np.random.randint(low=0, high=2, size=(560, 560))
   np.save("fake_img.npy", fake_img)
   np.save("fake_target.npy", fake_target)
   np.save("fake_mask.npy", fake_mask)
   print("fake_mask:", fake_mask)


if __name__ == "__main__":
    # gen_fake_data()
    # gen_fake_label()
    gen_fake_data2()