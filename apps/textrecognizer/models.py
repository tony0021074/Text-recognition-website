from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.db import models
import os

class OverwriteStorage(FileSystemStorage):
    '''
    Muda o comportamento padrão do Django e o faz sobrescrever arquivos de
    mesmo nome que foram carregados pelo usuário ao invés de renomeá-los.
    '''
    def get_available_name(self, name, max_length=None):
        if self.exists(name):
            os.remove(os.path.join(settings.MEDIA_ROOT, name))
        return name

class UploadImageModel(models.Model):
    def __str__(self):
        return self.question_text


    name = models.CharField(max_length=200, primary_key=True)
    image = models.ImageField(upload_to='uploadedImages', storage=OverwriteStorage())