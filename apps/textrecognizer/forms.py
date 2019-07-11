from django.forms import ModelForm
from .models import UploadImageModel

class UploadImageForm(ModelForm):
	'''
	def __init__(self, name=None, *args, **kwargs):
		super(UploadedImageForm, self).__init__(*args, **kwargs)
		self.fields['name'].value = name
	'''
	class Meta:
		model = UploadImageModel
		fields = ['image',]
		labels = {'image': '',}