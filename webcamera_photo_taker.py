import cv2
import os

cam = cv2.VideoCapture(0)

cv2.namedWindow("Photo taker")

def get_next_filename():
    if not os.path.isdir("test_images"):
        os.mkdir("test_images")

    test_images = os.listdir("test_images")
    if len(test_images) == 0:
        return "test_images/test_image_0001.png"
    test_images.sort()
    last = test_images[-1]
    number = int(last.removeprefix("test_image_").removesuffix(".png"))
    
    return f"test_images/test_image_{(number + 1):04d}.png"


while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Photo taker", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        # q pressed
        print("Q hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = get_next_filename()
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

cam.release()

cv2.destroyAllWindows()