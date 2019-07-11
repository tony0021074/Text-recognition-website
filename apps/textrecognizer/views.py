from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from .forms import UploadImageForm
from .models import UploadImageModel
import os
import cv2

from .src import textdetection as td
from .src import ocr

def index(request):
	if request.method == "POST":
		form = UploadImageForm(request.POST, request.FILES)
		if form.is_valid():
			# commit=False tells Django that "Don't send this to database yet.
			image = form.save(commit=False)
			image.name = request.FILES['image'].name
			image.save()
			return HttpResponseRedirect(reverse('textrecognizer:results', args=(image.name,)))
	else:
		form = UploadImageForm()
	return render(request, 'textrecognizer/index.html', {'form': form})

def results(request, imageName):
	image = get_object_or_404(UploadImageModel, pk=imageName)
	imagePath = os.path.abspath(image.image.path)
	markedFrame, texts, boxes = next(recognize(imagePath))
	cv2.imwrite(imagePath, markedFrame)
	textboxes = [{"text": text, "box": box, } for text, box in zip(texts, boxes)]
	return render(request, 'textrecognizer/results.html', {'image': image, 'textboxes': textboxes})


def recognize(filePath):
	eastModel = td.EASTModel(modelPath="apps/textrecognizer/src/resources/frozen_east_text_detection.pb")
	ocrModel = ocr.OCRModel()
	for markedFrame, textImgs, boxes in eastModel.predict(filePath):
		texts = []
		for textImg in textImgs:
			textImg = td.denoisingColored(textImg)
			textImg = td.toGrayImg(textImg)
			textImg = td.adaptiveThreshold(textImg)
			text = ocrModel.predict(textImg)
			texts.append(text)
		yield markedFrame, texts, boxes