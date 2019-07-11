import cv2
import math
import numpy as np
import copy
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EASTModel():

	def __init__(self, modelPath, inpWidth=320, inpHeight=320,
				 confThreshold=0.5, nmsThreshold=0.4):
		self.inpWidth = inpWidth
		self.inpHeight = inpHeight
		self.confThreshold = confThreshold
		self.nmsThreshold = nmsThreshold
		self.net = cv2.dnn.readNet(modelPath)
		self.outNames = []
		self.outNames.append("feature_fusion/Conv_7/Sigmoid")
		self.outNames.append("feature_fusion/concat_3")

	def predict(self, filePath):

		cap = cv2.VideoCapture(filePath)

		while cv2.waitKey(1) < 0:
			# Read frame
			hasFrame, frame = cap.read()
			markedFrame = copy.deepcopy(frame)
			if not hasFrame:
				cv2.waitKey()
				break

			# Get frame height and width
			height = frame.shape[0]
			width = frame.shape[1]
			rW = width / float(self.inpWidth)
			rH = height / float(self.inpHeight)

			# Create a 4D blob from frame.
			blob = cv2.dnn.blobFromImage(frame, 1.0, (self.inpWidth, self.inpHeight), (123.68, 116.78, 103.94), True, False)

			# Run the model
			self.net.setInput(blob)
			outs = self.net.forward(self.outNames)
			t, _ = self.net.getPerfProfile()
			label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

			# Get scores and geometry
			scores = outs[0]
			geometry = outs[1]
			(rects, confidences) = decode(scores, geometry, self.confThreshold)

			# Apply NMS
			indices = cv2.dnn.NMSBoxesRotated(rects, confidences, self.confThreshold, self.nmsThreshold)
			textImgs = []
			boxes = []
			for i in indices:
				# Get 4 corners of the rotated rect
				vertices = cv2.boxPoints(rects[i[0]])
				# Scale the bounding box coordinates based on the respective ratios
				for j in range(4):
					vertices[j][0] *= rW
					vertices[j][1] *= rH
				boxes.append([(round(point[0]), round(point[1])) for point in vertices])

				# Mark a box on the frame
				markImg(markedFrame, vertices)

				# Rotate the frame
				angle = rects[i[0]][2]
				rotatedFrame, m = rotate_image(frame, angle)

				# Rotate the box
				rotatedVertices = cv2.transform(np.array([vertices]), m)[0]
				x1, y1 = np.ceil(np.amax(rotatedVertices, axis=0))
				x2, y2 = np.floor(np.amin(rotatedVertices, axis=0))

				# Crop the frame
				textImg = rotatedFrame[int(y2.item()):int(y1.item()), int(x2.item()):int(x1.item())]

				textImgs.append(textImg)

			yield (markedFrame, textImgs, boxes)


def rotate_image(img, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = img.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    m = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(m[0,0])
    abs_sin = abs(m[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    m[0, 2] += bound_w/2 - image_center[0]
    m[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotatedImg = cv2.warpAffine(img, m, (bound_w, bound_h), borderMode=cv2.BORDER_WRAP)
    return rotatedImg, m


# Draw line on images following given points
def markImg(img, vertices):
	length = len(vertices)
	for j in range(length):
		startPoint = (vertices[j][0], vertices[j][1])
		endPoint = (vertices[(j + 1) % length][0], vertices[(j + 1) % length][1])
		cv2.line(img, startPoint, endPoint, (0, 255, 0), 1)


def denoisingColored(img):
	return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


# Adaptive Gaussian Thresholding
#img must be in gray scale
def adaptiveThreshold(img):
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								cv2.THRESH_BINARY,11,2)
	return img


def toGrayImg(img):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	return img


def decode(scores, geometry, scoreThresh):
	detections = []
	confidences = []

	############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
	assert len(scores.shape) == 4, "Incorrect dimensions of scores"
	assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
	assert scores.shape[0] == 1, "Invalid dimensions of scores"
	assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
	assert scores.shape[1] == 1, "Invalid dimensions of scores"
	assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
	assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
	assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
	height = scores.shape[2]
	width = scores.shape[3]
	for y in range(0, height):

		# Extract data from scores
		scoresData = scores[0][0][y]
		x0_data = geometry[0][0][y]
		x1_data = geometry[0][1][y]
		x2_data = geometry[0][2][y]
		x3_data = geometry[0][3][y]
		anglesData = geometry[0][4][y]
		for x in range(0, width):
			score = scoresData[x]

			# If score is lower than threshold score, move to next x
			if (score < scoreThresh):
				continue

			# Calculate offset
			offsetX = x * 4.0
			offsetY = y * 4.0
			angle = anglesData[x]

			# Calculate cos and sin of angle
			cosA = math.cos(angle)
			sinA = math.sin(angle)
			h = x0_data[x] + x2_data[x]
			w = x1_data[x] + x3_data[x]

			# Calculate offset
			offset = (
				[offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

			# Find points for rectangle
			p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
			p3 = (-cosA * w + offset[0], sinA * w + offset[1])
			center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
			detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
			confidences.append(float(score))

	# Return detections and confidences
	return (detections, confidences)