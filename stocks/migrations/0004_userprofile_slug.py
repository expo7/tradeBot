# Generated by Django 5.1 on 2024-08-10 17:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stocks', '0003_sector_slug'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='slug',
            field=models.SlugField(default='', max_length=200, unique=True),
        ),
    ]
