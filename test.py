from ultralytics import YOLO
model = YOLO(r"C:\Users\Kemji\Documents\BantAI trainer\runs\detect\BantAI\Model_v1\weights\best.pt")
results = model.predict(r"archive\Open-Flame-Hazards-YOLO\val\images\img_0013_13997cbe9e85.jpg")
results[0].show()