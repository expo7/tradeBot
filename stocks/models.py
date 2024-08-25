from django.db import models
from django.contrib.auth.models import User
from django.utils.text import slugify
from django.db.models.signals import post_save
from django.dispatch import receiver
from meta.models import ModelMeta
from django.urls import reverse
from django.db import models

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()
    
    


class TradingResult(models.Model):
    name = models.CharField(max_length=100, default='EquityGuardian')
    stock = models.CharField(max_length=10)
    backtest_plot = models.ImageField(upload_to='backtest_plots/', blank=True, null=True)
    start_date = models.DateField()
    end_date = models.DateField()
    profit = models.DecimalField(max_digits=15, decimal_places=2)
    buyhold_profit = models.DecimalField(max_digits=15, decimal_places=2)
    profit_per_day = models.DecimalField(max_digits=10, decimal_places=2)
    buyhold_per_day = models.DecimalField(max_digits=10, decimal_places=2)
    return_p = models.DecimalField(max_digits=6, decimal_places=4)
    return_bh = models.DecimalField(max_digits=6, decimal_places=4)
    return_daily = models.DecimalField(max_digits=6, decimal_places=4)
    return_bh_daily = models.DecimalField(max_digits=6, decimal_places=4)
    alpha = models.DecimalField(max_digits=6, decimal_places=4)
    beta = models.DecimalField(max_digits=6, decimal_places=4)
    sharpe_ratio = models.DecimalField(max_digits=6, decimal_places=4)
    sortino_ratio = models.DecimalField(max_digits=6, decimal_places=4)
    in_market_days = models.IntegerField()
    total_days = models.IntegerField()

    def __str__(self):
        return f"{self.stock} - Profit: {self.profit}"


class Tag(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Article(ModelMeta, models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True, max_length=200, default='')
    content = models.TextField()
    author = models.CharField(max_length=100, default='unknown Author')
    published_date = models.DateTimeField(auto_now_add=True)
    tags = models.ManyToManyField(Tag, blank=True)

    _metadata = {
        'title': 'title',
        'description': 'content',
        'keywords': 'get_keywords',
        'author': 'author',
        'published_time': 'published_date',
    }

    def get_keywords(self):
        return ', '.join(tag.name for tag in self.tags.all())

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title
    def get_absolute_url(self):
        return reverse('article_detail', args=[self.slug])

class Sector(ModelMeta, models.Model):
    slug = models.SlugField(unique=True, max_length=200, default='')
    name = models.CharField(max_length=100)
    articles = models.ManyToManyField(Article)

    _metadata = {
        'title': 'name',
        'description': 'get_description',
        'keywords': 'get_keywords',
    }

    def get_description(self):
        return f"Analysis and articles about the {self.name} sector."

    def get_keywords(self):
        return ', '.join(article.title for article in self.articles.all())

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
    def get_absolute_url(self):
        return reverse('sector_detail', args=[self.slug])

class Stock(ModelMeta, models.Model):
    ticker = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    sector = models.ForeignKey(Sector, on_delete=models.CASCADE)
    revenue = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    net_income = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)

    _metadata = {
        'title': 'name',
        'description': 'get_description',
        'keywords': 'ticker',
    }

    def get_description(self):
        return f"Financial metrics and analysis for {self.name} ({self.ticker})."

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.ticker)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.ticker

    def get_absolute_url(self):
        return reverse('stock_detail', args=[self.slug])

class UserProfile(ModelMeta, models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    slug = models.SlugField(unique=True, max_length=200, default='')
    favorite_stocks = models.ManyToManyField(Stock)

    _metadata = {
        'title': 'user.username',
        'description': 'get_description',
    }

    def get_description(self):
        return f"Profile of {self.user.username} with favorite stocks."

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.user.username)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.user.username

# from django.db import models
# from django.contrib.auth.models import User
# from django.utils.text import slugify
# from django.db.models.signals import post_save
# from django.dispatch import receiver
# from meta.models import ModelMeta

# @receiver(post_save, sender=User)
# def create_user_profile(sender, instance, created, **kwargs):
#     if created:
#         UserProfile.objects.create(user=instance)

# @receiver(post_save, sender=User)
# def save_user_profile(sender, instance, **kwargs):
#     instance.profile.save()

# class Tag(models.Model):
#     name = models.CharField(max_length=100)

#     def __str__(self):
#         return self.name

# class Article(ModelMeta, models.Model):
#     title = models.CharField(max_length=200)
#     slug = models.SlugField(unique=True, max_length=200, default='')
#     content = models.TextField()
#     author = models.CharField(max_length=100, default='unknown Author')
#     published_date = models.DateTimeField(auto_now_add=True)
#     tags = models.ManyToManyField(Tag, blank=True)

#     _metadata = {
#         'title': 'title',
#         'description': 'content',
#         'keywords': 'get_keywords',
#         'author': 'author',
#         'published_time': 'published_date',
#     }

#     def get_keywords(self):
#         return ', '.join(tag.name for tag in self.tags.all())

#     def save(self, *args, **kwargs):
#         if not self.slug:
#             self.slug = slugify(self.title)
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return self.title

# class Sector(ModelMeta, models.Model):
#     slug = models.SlugField(unique=True, max_length=200, default='')
#     name = models.CharField(max_length=100)
#     articles = models.ManyToManyField(Article)

#     _metadata = {
#         'title': 'name',
#         'description': 'get_description',
#         'keywords': 'get_keywords',
#     }

#     def get_description(self):
#         return f"Analysis and articles about the {self.name} sector."

#     def get_keywords(self):
#         return ', '.join(article.title for article in self.articles.all())

#     def save(self, *args, **kwargs):
#         if not self.slug:
#             self.slug = slugify(self.name)
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return self.name

# class Stock(ModelMeta, models.Model):
#     ticker = models.CharField(max_length=10, unique=True)
#     name = models.CharField(max_length=100)
#     sector = models.ForeignKey(Sector, on_delete=models.CASCADE)
#     revenue = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
#     net_income = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)

#     _metadata = {
#         'title': 'name',
#         'description': 'get_description',
#         'keywords': 'ticker',
#     }

#     def get_description(self):
#         return f"Financial metrics and analysis for {self.name} ({self.ticker})."

#     def save(self, *args, **kwargs):
#         if not self.slug:
#             self.slug = slugify(self.ticker)
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return self.ticker

# class UserProfile(ModelMeta, models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
#     slug = models.SlugField(unique=True, max_length=200, default='')
#     favorite_stocks = models.ManyToManyField(Stock)

#     _metadata = {
#         'title': 'user.username',
#         'description': 'get_description',
#     }

#     def get_description(self):
#         return f"Profile of {self.user.username} with favorite stocks."

#     def save(self, *args, **kwargs):
#         if not self.slug:
#             self.slug = slugify(self.user.username)
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return self.user.username




    
# class Tag(models.Model):
#     name = models.CharField(max_length=100)

#     def __str__(self):
#         return self.name

# class Article(models.Model):
#     title = models.CharField(max_length=200)
#     slug = models.SlugField(unique=True, max_length=200,default='')
#     content = models.TextField()
#     author = models.CharField(max_length=100,default='unknown Author')
#     published_date = models.DateTimeField(auto_now_add=True)
#     # category = models.ForeignKey(Category, on_delete=models.CASCADE)
#     tags = models.ManyToManyField(Tag, blank=True)

#     def save(self, *args, **kwargs):
#         if not self.slug:
#             self.slug = slugify(self.title)
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return self.title


    
    
# class Sector(models.Model):
#     slug = models.SlugField(unique=True, max_length=200,default='')
#     name = models.CharField(max_length=100)
#     articles = models.ManyToManyField(Article)
#     def save(self, *args, **kwargs):
#         if not self.slug:
#             self.slug = slugify(self.name)
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return self.name
    
    
# class Stock(models.Model):
#     ticker = models.CharField(max_length=10, unique=True)
#     name = models.CharField(max_length=100)
#     sector = models.ForeignKey(Sector, on_delete=models.CASCADE)
#     revenue = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
#     net_income = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
#     # Add other financial metrics as needed
    
#     def save(self, *args, **kwargs):
#         if not self.slug:
#             self.slug = slugify(self.ticker)
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return self.ticker
    
    
# class UserProfile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE,related_name='profile')
#     slug = models.SlugField(unique=True, max_length=200,default='')

#     favorite_stocks = models.ManyToManyField(Stock)
#     # Add other profile fields if needed
#     def save(self, *args, **kwargs):
#         if not self.slug:
#             self.slug = slugify(self.user.username)
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return self.user.username
        


