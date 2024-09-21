import cv2
import numpy as np

def process_image(image_path, output_path="blurred_output.jpg"):
    # تحميل الصورة
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("الصورة لم تُحمل بنجاح. تحقق من المسار.")

    # تحويل الصورة إلى تدرجات الرمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # تحميل مصنف الوجه
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError("مصنف الوجه لم يُحمّل بنجاح. تحقق من المسار.")

    # تحديد الوجوه في الصورة
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("لم يتم العثور على أي وجوه في الصورة.")

    # تغويش الصورة بالكامل
    blurred_image = cv2.GaussianBlur(image, (25, 25), 40)

    # زيادة حجم الإطار حول الوجه
    padding = 20

    # دمج الوجه الأصلي مع الصورة المغوشة
    for (x, y, w, h) in faces:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        blurred_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    # حفظ الصورة النهائية
    cv2.imwrite(output_path, blurred_image)
    print(f"تم حفظ الصورة النهائية في: {output_path}")

   # عرض الصورة النهائية
    cv2.imshow("Blurred Image", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # إدخال المسار المطلوب للصورة 
    input_image_path = "id3.jpg"
    process_image(input_image_path)
