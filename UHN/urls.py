from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload'),
    path('get_server_messages/', views.get_server_messages, name='get_server_messages'),
    path('check_exists/<name>', views.check_exists, name = 'check_exists'),
    path('delete_file/', views.delete_file, name='delete_file'),
    path('sort_by/<order_by>', views.sort_by, name='sort_by')
]