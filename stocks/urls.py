from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView
from django.contrib.sitemaps.views import sitemap
from django.urls import path
from .sitemaps import ArticleSitemap, SectorSitemap, StockSitemap
from django.views.generic import TemplateView

sitemaps = {
    'articles': ArticleSitemap,
    'sectors': SectorSitemap,
    'stocks': StockSitemap,
}



urlpatterns = [
    path('', views.home, name='home'),
    path('sector/<slug:slug>/', views.sector_detail, name='sector_detail'),
    # path('sector_analysis/', views.sector_analysis, name='sector_analysis'),
    path('article/<slug:slug>/', views.article_detail, name='article_detail'),
    path('trade/', views.trade_list, name='trade_list'),
    path('trade/<int:id>/', views.trade_detail, name='trade_detail'),
    path('articles/', views.article_list, name='article_list'),  # Ensure this pattern exists
    path('login/', views.login_view.as_view(), name='login'),
    path('register/', views.register_view, name='register'),
    # path('profile/', views.user_profile, name='user_profile'),
    path('logout/', views.logout_view, name='logout'),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}, name='django.contrib.sitemaps.views.sitemap'),
    path('robots.txt', TemplateView.as_view(template_name="robots.txt", content_type="text/plain")),


]