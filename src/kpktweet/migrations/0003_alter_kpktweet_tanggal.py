# Generated by Django 4.0 on 2021-12-18 03:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('kpktweet', '0002_alter_kpktweet_tanggal'),
    ]

    operations = [
        migrations.AlterField(
            model_name='kpktweet',
            name='tanggal',
            field=models.TextField(),
        ),
    ]
