import pytesseract

class OCRModel():
	def __init__(self, config=("-l eng --oem 1 --psm 8")):
		self.config=config

	def predict(self, img):
		return pytesseract.image_to_string(img, config=self.config)