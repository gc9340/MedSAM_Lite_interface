# Generated by Django 5.0.6 on 2024-05-28 21:24

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("UHN", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="testfile",
            name="uploaded_at",
            field=models.DateTimeField(
                auto_now_add=True, default=django.utils.timezone.now
            ),
            preserve_default=False,
        ),
    ]
