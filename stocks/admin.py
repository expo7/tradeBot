from django.contrib import admin
from .models import Stock, Sector, Article, UserProfile, TradingResult




# Register models with admin
admin.site.register(Stock)
admin.site.register(Sector)
admin.site.register(Article)
admin.site.register(UserProfile)
admin.site.register(TradingResult)
# Optional: Define and register custom admin classes
class StockAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'name', 'sector', 'revenue', 'net_income')
    search_fields = ('ticker', 'name')
    list_filter = ('sector',)

class ArticleAdmin(admin.ModelAdmin):
    list_display = ('title', 'sector', 'published_date')
    search_fields = ('title', 'content')
    list_filter = ('sector',)

# Ensure you don't register Stock again
# admin.site.register(Stock, StockAdmin)  # This would cause the error if it's already registered
