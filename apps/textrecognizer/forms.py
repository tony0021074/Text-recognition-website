from django.forms import ModelForm
from .models import UploadedImageModel

class UploadedImageForm(ModelForm):
	'''
	def __init__(self, name=None, *args, **kwargs):
		super(UploadedImageForm, self).__init__(*args, **kwargs)
		self.fields['name'].value = name
	'''
	class Meta:
		model = UploadedImageModel
		fields = ['uploadedImage',]
		labels = {'uploadedImage': '',}