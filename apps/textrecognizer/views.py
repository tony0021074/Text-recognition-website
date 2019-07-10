from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from .forms import UploadedImageForm
from .models import UploadedImageModel
import os
import cv2

from .src import textdetection as td
from .src import ocr

from PIL import Image


def index(request):
	if request.method == "POST":
		form = UploadedImageForm(request.POST, request.FILES)
		if form.is_valid():
			# commit=False tells Django that "Don't send this to database yet.
			uploadedImage = form.save(commit=False)
			uploadedImage.name = request.FILES['uploadedImage'].name
			uploadedImage.save()
			return HttpResponseRedirect(reverse('textrecognizer:results', args=(uploadedImage.name,)))
	else:
		form = UploadedImageForm()
	return render(request, 'textrecognizer/index.html', {'form': form})


def results(request, imageName):
	image = get_object_or_404(UploadedImageModel, pk=imageName)
	imagePath = os.path.abspath(image.uploadedImage.path)
	markedFrame, texts, vertices = next(recognize(imagePath))
	cv2.imwrite(imagePath, markedFrame)
	return render(request, 'textrecognizer/results.html', {'image': image,})


def recognize(filePath):
	eastModel = td.EASTModel(modelPath="apps/textrecognizer/src/resources/frozen_east_text_detection.pb")
	ocrModel = ocr.OCRModel()
	for markedFrame, textImgs, vertices in eastModel.predict(filePath):
		texts = []
		for textImg in textImgs:
			textImg = td.denoisingColored(textImg)
			textImg = td.toGrayImg(textImg)
			textImg = td.adaptiveThreshold(textImg)
			text = ocrModel.predict(textImg)
			texts.append(text)
		yield markedFrame, texts, vertices