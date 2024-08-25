from django.contrib.sitemaps import Sitemap
from django.urls import reverse
from .models import Article, Sector, Stock

class ArticleSitemap(Sitemap):
    changefreq = "weekly"
    priority = 0.8

    def items(self):
        return Article.objects.all()

    def lastmod(self, obj):
        return obj.published_date

class SectorSitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.6

    def items(self):
        return Sector.objects.all()

class StockSitemap(Sitemap):
    changefreq = "daily"
    priority = 0.9

    def items(self):
        return Stock.objects.all()
# from django.contrib.sitemaps import Sitemap
# from django.urls import reverse

# class StaticViewSitemap(Sitemap):
#     priority = 0.5
#     changefreq = 'daily'

#     def items(self):
#         # return ['home', 'about', 'contact']
#         return ['home', 'article_list', 'login', 'register', 'user_profile', 'logout', 'sector_detail', 'sector_analysis', 'article_detail']

#     def location(self, item):
#         return reverse(item)