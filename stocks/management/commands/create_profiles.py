from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from stocks.models import UserProfile

class Command(BaseCommand):
    help = 'Create UserProfile for existing users'

    def handle(self, *args, **kwargs):
        users = User.objects.all()
        for user in users:
            if not UserProfile.objects.filter(user=user).exists():
                UserProfile.objects.create(user=user)
                self.stdout.write(self.style.SUCCESS(f'Created profile for user: {user.username}'))
            else:
                self.stdout.write(self.style.WARNING(f'Profile already exists for user: {user.username}'))