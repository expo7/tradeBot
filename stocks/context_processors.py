from .models import Sector

def sectors(request):
    return {
        'sectors': Sector.objects.all()
    }