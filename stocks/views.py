from django.shortcuts import render, get_object_or_404,redirect
"""
This module contains views for the stocks app.
Functions:
- home(request): Renders the home page with a list of sectors and a specific article.
- stock_list(request): Renders the stock list page with all stocks.
- article_list(request): Renders the article list page with all articles and sectors.
- article_detail(request, slug): Renders the article detail page for a specific article.
- sector_detail(request, slug): Renders the sector detail page for a specific sector.
- login_view(FormView): Renders the login page and handles user authentication.
- register_view(request): Renders the registration page and handles user registration.
- user_profile(request): Renders the user profile page (requires login).
- logout_view(request): Logs out the user and redirects to the home page.
"""
# from django.contrib.auth import login, authenticate, logout
# from django.contrib.auth.decorators import login_required
# from .models import Stock, Sector, Article
# from .forms import LoginForm, RegisterForm  # Assuming you have these forms defined
# from django.views.generic import FormView
# from django.contrib.auth.forms import AuthenticationForm
# from django.contrib.auth import login as auth_login
# from .models import Sector
# from django.views.generic.base import TemplateView


from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages  # Import messages framework
from .models import Stock, Sector, Article, TradingResult
from .forms import LoginForm, RegisterForm  # Assuming you have these forms defined
from django.views.generic import FormView
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login
from django.views.generic.base import TemplateView

# Login view
class login_view(FormView):
    template_name = 'stocks/login.html'
    form_class = AuthenticationForm

    def form_valid(self, form):
        user = form.get_user()
        auth_login(self.request, user)
        return redirect('home')

    def form_invalid(self, form):
        messages.error(self.request, 'Invalid username or password.')
        return self.render_to_response(self.get_context_data(form=form))

# Register view
def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Registration failed. Please correct the errors below.')
    else:
        form = RegisterForm()
    return render(request, 'stocks/register.html', {'form': form})

# Other views...

class RobotsView(TemplateView):
    template_name = "robots.txt"
    content_type = "text/plain"

def home(request):
    """
    Renders the home page of the stocks app.

    Parameters:
    - request: The HTTP request object.

    Returns:
    - A rendered HTML template of the home page with the following context variables:
      - sectors: A queryset of all sector models.
      - article: The article object with the slug 'What-is-Fundamental-Analysis'.

    Raises:
    - Http404: If the article with the specified slug does not exist.
    """
    sector_models = Sector.objects.all()
    article = get_object_or_404(Article, slug='What-is-Fundamental-Analysis')
    return render(request, 'stocks/home.html', {'sectors': sector_models,'article': article})
def trade_detail(request, id):
    trade = get_object_or_404(TradingResult, id=id)
    return render(request, 'stocks/trade_detail.html', {'trade': trade})
def trade_list(request):
    results = TradingResult.objects.all()
    return render(request, 'stocks/trade_list.html', {'results': results})
# def trade_detail(request, id):
#     trade = get_object_or_404(TradingResult, id=id)
#     return render(request, 'stocks/trade_detail.html', {'trade': trade})


def stock_list(request):
    stocks = Stock.objects.all()
    return render(request, 'stocks/stock_list.html', {'stocks': stocks})

def article_list(request):
    sector_models = Sector.objects.all()
    articles = Article.objects.all()
    return render(request, 'stocks/article_list.html', {'articles': articles,'sectors': sector_models})

def article_detail(request, slug):
    sector_models = Sector.objects.all()
    article = get_object_or_404(Article, slug=slug)
    return render(request, 'stocks/article_detail.html', {'article': article,'sectors': sector_models})

    

def sector_detail(request, slug):
    if slug=='Technology':
        article = get_object_or_404(Article, slug='Understanding-Fundamental-Analysis-in-the-Tech-Sector')
    if slug=='Finance':
        article = get_object_or_404(Article, slug='Understanding-Fundamental-Analysis-in-the-Finance-Sector')
    sector = get_object_or_404(Sector, slug=slug)
    related_articles = sector.articles.all()
    sector_models = Sector.objects.all()
    return render(request, 'stocks/sector_detail.html', {'sector': sector, 'article': article,'related_articles': related_articles,'sectors': Sector.objects.all()})

# class login_view(FormView):
#     template_name = 'stocks/login.html'
#     form_class = AuthenticationForm
    
#     def form_valid(self, form):
#         auth_login(self.request, form.get_user())
#         return redirect('home')  # Redirect to home or any other page after login


# def register_view(request):
#     if request.method == 'POST':
#         form = RegisterForm(request.POST)
#         if form.is_valid():
#             user = form.save()
#             login(request, user)
#             return redirect('home')
#     else:
#         form = RegisterForm()
#     return render(request, 'stocks/register.html', {'form': form})

# @login_required
# def user_profile(request):
#     return render(request, 'stocks/user_profile.html')

def logout_view(request):
    logout(request)
    return redirect('home')
