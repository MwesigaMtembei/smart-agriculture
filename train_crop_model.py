import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib
from sklearn.metrics import accuracy_score, hamming_loss, classification_report

# -------------------------------
# 1. Flatten Raw Data
# -------------------------------

raw_data = [
    {
       "region": "Arusha",
        "district": ["Ngorongoro", "Longido", "Karatu"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi", "Maharage", "Mtama", "Uwele", "Ngano", "Shayiri", "Chai", "Parachichi"]
    },
    {
        "region": "Arusha",
        "district": ["Ngorongoro", "Longido", "Karatu", "Monduli", "Arumeru"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi", "Maharage", "Ngano", "Shayiri", "Mtama", "Uwele", "Kahawa", "Chai", "Migomba", "Viazi", "Alizeti", "Parachichi"]
    },
    {
        "region": "Arusha",
        "district": ["Ngorongoro", "Longido", "Karatu", "Monduli", "Arumeru"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi", "Maharage", "Ngano", "Shayiri", "Mtama", "Uwele", "Kahawa", "Chai", "Migomba", "Viazi", "Alizeti", "Parachichi"]
    },
    {
        "region": "Arusha",
        "district": ["Ngorongoro", "Longido", "Karatu", "Monduli", "Arumeru"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Mahindi", "Maharage", "Ngano", "Shayiri", "Mtama", "Uwele", "Kahawa", "Chai", "Migomba", "Viazi", "Alizeti", "Mboga mboga"]
    },
    {
        "region": "Arusha",
        "district": ["Ngorongoro", "Longido", "Monduli", "Arumeru"],
        "ph_min": 8.5,
        "ph_max": 14.0,
        "acidity_level": "Highly Alkaline",
        "recommended_crop": ["Mahindi", "Maharage", "Mtama", "Uwele", "Viazi", "Alizeti", "Mboga mboga"]
    },
    
     {
        "region": "Dodoma",
        "district": ["Mpwapwa","Kongwa","Bahi","Chamwino","Chemba","Kondoa"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Zabibu","Uwele","Alizeti","Muhogo","Viazi Vitamu","Mtama","Karanga","Muhogo","Korosho","Ufuta"]
    },
     {
        "region": "Dodoma",
        "district": ["Chemba","Kondoa","Mpwapwa","Kongwa","Bahi","Chamwino"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Zabibu","Maharage","Uwele","Ulezi","Alizeti","Muhogo","Viazi Vitamu","Mtama","Karanga","Korosho","Ufuta"]
    },
     {
        "region": "Dodoma",
        "district": ["Chamwino","Bahi","Kongwa","Kondoa","Mpwapwa","Chemba"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Uwele","Ulezi","Alizeti","Muhogo","Viazi","Mtama","Karanga","Korosho","Ufuta"]
    },
     {
        "region": "Dodoma",
        "district": ["Kondoa","Bahi","Chamwino"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Uwele","Ulezi","Alizeti","Viazi"]
    },
    {
        "region": "Geita",
        "district": ["Bukombe","Chato","Geita","Nyang`hwale"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Pamba"," Muhogo"," Viazi Vitamu","Mpunga","Mahindi","Mtama","Mananasi","Karanga","Dengu"]
    },
    {
        "region": "Geita",
        "district": ["Bukombe,Mbogwe,Chato,Geita,Nyang`hwale"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Pamba", "Muhogo"," Viazi Vitamu","Mpunga","Mahindi","Mtama","Mananasi","Karanga","Kahawa,Dengu"]
    },
    {
        "region": "Geita",
        "district": ["Mbogwe","Nyang`hwale"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Pamba", "Muhogo", "Viazi Vitamu","Mpunga","Mahindi","Mtama","Mananasi","Karanga","Kahawa","Dengu"]
    },
     {
        "region": "Iringa",
        "district": ["Mufindi","Kilolo","Iringa"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Chai","Maharage","Alizeti","Muhogo","Viazi,Tumbaku"]
    },
     {
        "region": "Iringa",
        "district": ["Iringa DC","kilolo","Mufindi"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Maharage","Pareto","Njegere,Viazi","Tumbaku"]
    },
     {
        "region": "Iringa",
        "district": ["Iringa DC","Kilolo"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Njegere","Mbogamboga","Migomba","Alizeti","Viazi"]
    },
     {
        "region": "Iringa ",
        "district": ["Iringa DC"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Mbogamboga","Alizeti","Viazi"]
    },
    {
        "region": "Kagera",
        "district": ["Misenyi","Bukoba Rural","Muleba","Biharamulo","Ngara","Karagwe","Kyerwa"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Muhogo","Mtama"",Maharage","Migomba","Kahawa","Chai","Miwa","Pamba","Parachichi","Vanila","Mbogamboga"]
    },
    {
        "region": "Kagera",
        "district": ["Bukoba Rural","Muleba","Biharamulo","Ngara","Karagwe","Kyerwa","Misenyi"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Muhogo"",Mtama","Maharage","Migomba","Kahawa","Chai","Miwa","Pamba","Parachichi","Vanila","Mbogamboga"]
    },
    {
        "region": "Kagera",
        "district": ["Misenyi"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Maharage","Migomba","Kahawa","Miwa,Mbogamboga"]
    },
    {
        "region": "Katavi",
        "district": ["Tanaganyika","Nsimbo,Mlele"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Muhogo", "Viazi Vitamu","Mahindi","Ufuta","Alizeti"]
    },
    {
        "region": "Katavi",
        "district": ["Mpanda","Mpimbwe","Nsimbo"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Ufuta","Muhogo","Karanga","Minazi","Ulezi","Alizeti","Viazi Vitamu"]
    },
    {
        "region": "Katavi",
        "district": ["Mpanda","Mpimbwe"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Ufuta","Muhogo","Karanga","Minazi","Ulezi","Alizeti","Viazi Vitamu"]
    },
     {
        "region": "Kilimanjaro",
        "district": ["Same","Mwanga"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi", "Mtama", "Uwele","Viazi Vitamu","Mkonge","Muhogo","Tangawizi", "Parachichi"]
    },
     {
        "region": "Kilimanjaro",
        "district": ["Same","Mwanga","Moshi Rural","Rombo","Hai","Siha"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Maharage", "Mtama", "Uwele","Kahawa","Chai","Migomba","Viazi" ,"Alizeti","Mkonge","Muhogo","Tangawizi","Parachichi"]
    },
     {
        "region": "Kilimanjaro",
        "district": ["Same","Mwanga","Moshi Rural","Rombo","Hai","Siha"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Maharage", "Mtama", "Uwele","Kahawa","Chai","Migomba","Viazi" ,"Alizeti","Mkonge","Muhogo","Tangawizi","Parachichi"]
    },
     {
        "region": "Kilimanjaro",
        "district": ["Same","Mwanga","Moshi Rural","Rombo","Hai","Siha"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Mahindi","Mpunga" ,"Maharage", "Mtama", "Uwele","Migomba","Viazi ","Alizeti","Mbogamboga"]
    },
     {
        "region": "Kilimanjaro",
        "district": ["Same","Moshi Rural"],
        "ph_min": 8.5,
        "ph_max": 14.0,
        "acidity_level": "Highly Alkaline",
        "recommended_crop": ["Mahindi","Mpunga" ,"Maharage"," Mtama", "Uwele","Viazi" ,"Alizeti","Mbogamboga"]
    },
      {
        "region": "Manyara",
        "district": ["Kiteto","Simanjiro","Mbulu","Babati","Hanang"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi", "Maharage", "Mtama", "Ngano", "Shayiri","Ulezi","Viazi","Alizeti"]
    },
 {
        "region": "Manyara",
        "district": ["Kiteto","Simanjiro","Mbulu","Babati","Hanang"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi", "Maharage","Kunde", "Ngano"," Shayiri", "Mtama","Ulezi","Kahawa","Migomba","Maboga","Viazi","Alizeti","Mbaazi","Vitunguu"]
    },
 {
        "region": "Manyara",
        "district": ["Simanjiro","Mbulu","Babati","Hanang","Kiteto"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Kunde","Ngano","Maboga","Shayiri","Ulezi","kahawa","Migomba","Viazi","Alizeti","Mbaazi","Vitunguu"]
    },
 {
        "region": "Manyara",
        "district": ["Simanjiro","Mbulu","Babati","Kiteto"],
        "ph_min": 8.5,
        "ph_max": 14.0,
        "acidity_level": "Slightlly Alkaline",
        "recommended_crop": ["Mahindi", "Mpunga" ,"Maharage", "Ngano", "Shayiri", "Migomba", "Viazi", "Alizeti", "Mbogamboga","Vitunguu"]
    },
 {
        "region": "Manyara",
        "district": ["Simanjiro","Babati"],
        "ph_min": 8.5,
        "ph_max": 14.0,
        "acidity_level": "Highly Alkaline",
        "recommended_crop": ["Mahindi"," Mpunga ","Maharage", "Mtama","Uwele", "Viazi", "Alizeti", "Mbogamboga"]
    },
 {
        "region": "Mara",
        "district": ["Tarime","Rorya","Musoma Rural","Bunda","Serengeti"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Viazi vitamu","Tumbaku","Muhogo","Mtama","Maharage","Miwa","Machungwa","Dengu","Kunde","Choroko","Karanga","Njegere","Alizeti","Ufuta","Vitunguu"]
    },
     {
        "region": "Mara",
        "district": ["Serengeti","Bunda","Musoma Rural","Tarime","Rorya"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Viazi vitamu ","Muhogo","Mtama","Maharage","kahawa","Migomba","Miwa","Machungwa","Dengu","Kunde","Choroko","Karanga","Njegere","Alizeti","Ufuta","Vitunguu","Mbogamboga","Tumbaku"]
    },
     {
        "region": "Mara",
        "district": ["Serengeti","Bunda","Musoma Rural","Tarime,Rorya"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Viazi vitamu" ,"Muhogo","Mtama","Maharage","kahawa","Migomba","Miwa","Machungwa","Dengu","Kunde","Choroko","Karanga","Njegere"," Alizeti","Ufuta","Vitunguu","Mbogamboga"]
    },
     {
        "region": "Mara ",
        "district": ["Serengeti"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Pamba","Viazi vitamu","Mahindi","Mtama","Alizeti","Vitunguu","Mbogamboga"]
    },
    {

        "region": "Mbeya",
        "district": ["Chunya","Mbarali","Rungwe","Mbeya"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi", "Maharage","Uwele","Ulezi","Alizeti","Muhogo","Viazi","Mtama","Karanga","Tumbaku"]
    },
 {
        "region": "Mbeya",
        "district": ["Chunya","Mbarali","Rungwe","Mbeya"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi", "Maharage","Mpunga","Migomba","Kahawa","Chai","Hiliki","Uwele","Ngano","Alizeti","Muhogo","Viazi","Mtama","Karanga","Tumbaku","Pamba","Korosho","Matikiti","Kakao"]
    },
 {
        "region": "Mbeya",
        "district": ["Mbarali","Chunya","Mbeya","Kyela"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Migomba","Kahawa","Chai","Hilik","Uwele","Ngano","Alizeti","Muhogo","Viazi","Mtama","Karanga","Tumbaku","Pamba","Korosho","Matikiti","Kakao"]
    },
 {
        "region": "Mbeya",
        "district": ["Mbarali,Chunya,Mbeya"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Mahindi", "Mpunga" ,"Maharage"," Uwele","Alizeti","Viazi","Mtama","Karanga","Pamba"]
    },
 {
        "region": "Mbeya",
        "district": ["Mbeya"],
        "ph_min": 8.5,
        "ph_max": 14.0,
        "acidity_level": "Highly Alkaline",
        "recommended_crop": ["Mahindi"," Mpunga" ,"Maharage","Ngano", "Viazi","Migomba"]
    },

 {
        "region": "Morogoro",
        "district": ["Kilosa","Mvomero","Morogoro rural","Ulanga","Kilombero"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Maharage","Kahawa","Miwa","Viungo","Uwele","Alizeti","Muhogo","Viazi","Mkonge"]
    },
     {
        "region": "Morogoro",
        "district": ["Kilosa","Mvomero","Morogoro rural","Ulanga","Kilombero"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Njegere","Mbogamboga","Miwa","Migomba","Magimbi","Kahawa","Viungo","Alizeti","Muhogo","Viazi","Mkonge"]
    },
     {
        "region": "Morogoro",
        "district": ["Kilosa","Mvomero","Morogoro rural","Ulanga","Kilombero"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Miwa","Migomba","Kahawa","Viungo","Miembe","Alizeti","Viazi"]
    },
     {
        "region": "Morogoro ",
        "district": ["Morogoro Rural"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Migomba","Alizeti"]
    },
    
       { "region": "Mwanza",
        "district": ["Ukerewe","Sengerema","Magu"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Pamba","Muhogo","Viazi vitamu","Mpunga","Mahindi","Mtama","Mananasi","Karanga","Migomba","Chikichi","Kahawa","Ufuta", "Alizeti","Vanila","Njugu","Mbogamboga"]
    },
     {
        "region": "Mwanza",
        "district": ["Kwimba","Misungwi","Magu","Sengerema","Ukerewe"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Pamba","Muhogo","Viazi vitamu","Mpunga","Mahindi","Mtama","Mananasi","Karanga","Migomba","Chikichi","Kahawa","Ufuta" ,"Alizeti","Vanila","Njugu","Mbogamboga"]
    },
     {
        "region": "Mwanza",
        "district": ["Kwimba","Misungwi","Magu","Sengerema"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Pamba","Muhogo","Viazi vitamu","Mpunga","Mahindi","Mtama","Mananasi","Karanga", "Alizeti","Maharage","Kunde","Mbogamboga"]
    },
     {
        "region": "Mwanza ",
        "district": ["Misungwi","Magu"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Pamba","Viazi vitamu","Mpunga","Mahindi","Mtama","Mananasi","Karanga","Alizeti","Maharage","Kunde","Mbogamboga"]
    },

{
        "region": "Njombe",
        "district": ["Njombe","Ludewa","Makete","Wanging`ombe"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Maharage","Chai","Parachichi","Kahawa","Ngano","Alizeti","Viazi"]
    },
    {
        "region": "Njombe",
        "district": ["Njombe","Ludewa","Makete","Wanging`ombe"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Chai","Parachichi","Kahawa","Ngano","Alizeti","Viazi","Migomba"]
    },
    {
        "region": "Njombe",
        "district": ["Makete","Wanging`ombe"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Kahawa","Ngano","Alizeti","Viazi","Migomba","Njegere"]
    },
    
       { "region": "Rukwa",
        "district": ["Nkasi","Kalambo","Sumbawanga Rural","Sumbawanga"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Maharage","Uwele","Alizeti","Muhogo"]
    },
     {
        "region": "Rukwa",
        "district": ["Nkasi","Kalambo","Sumbawanga Rural","Sumbawanga"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Uwele","Ngano","Alizeti","Muhogo"]
    },
     {
        "region": "Rukwa",
        "district": ["Sumbawanga Rural"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Uwele","Alizeti"]
    },
     {
        "region": "Rukwa ",
        "district": ["Sumbawanga Rural"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Uwele","Alizeti"]
    },

{ "region": "Shinyanga",
        "district": ["Kahama"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Maharage","Uwele","Alizeti","Muhogo","Viazi vitamu","Mtama","Karanga","Njugu","Tumbaku"]
    },
     {
        "region": "Shinyanga",
        "district": ["Kahama ","Shinyanga Rural","Kishapu"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Maharage","Uwele","Ngano","Alizeti","Muhogo","Viazi vitamu","Mtama","Karanga","Njugu","Tumbaku","Pamba","Choroko","Dengu","Kunde"]
    },
     {
        "region": "Shinyanga",
        "district": ["Kahama ","Shinyanga Rural","Kishapu"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Uwele","Alizeti","Muhogo","Viazi vitamu","Mtama","Karanga","Pamba","Choroko","Dengu","Kunde"]
    },
     {
        "region": "Shinyanga ",
        "district": ["Kishapu"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Uwele","Alizeti","Viazi vitamu","Mtama","Pamba","Choroko","Dengu","Kunde"]
    },

{ 
    "region": "Simiyu",
        "district": ["Bariadi","Maswa","Meatu","Itilima","Busega"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Pamba","Muhogo","Viazi vitamu","Mpunga","Mahindi","Mtama","Uwele","Karanga","Alizeti"]
    },
     {
        "region": "Simiyu",
        "district": ["Bariadi","Maswa","Meatu","Itilima","Busega"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Pamba","Muhogo","Viazi vitamu","Mpunga","Mahindi","Mtama","Uwele","Karanga","Alizeti"]
    },
     {
        "region": "Simiyu",
        "district": ["Meatu"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Pamba","Viazi vitamu","Mpunga","Mahindi","Mtama","Uwele","Alizeti"]
    },
     {
        "region": "Simiyu ",
        "district": ["Meatu"],
        "ph_min": 8.5,
        "ph_max": 14.0,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Pamba","Viazi vitamu","Mpunga","Mahindi","Mtama","Uwele","Alizeti"]
    },

{
        "region": "Singida",
        "district": ["Manyoni","Itigi","Ikungi","Singida Rural","Iramba","Mkalama"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Uwele","Alizeti","Muhogo","Viazi vitamu","Mtama","Karanga","Tumbaku","Korosho"]
    },
 {
        "region": "Singida",
        "district": ["Manyoni","Itigi","Ikungi","Singida Rural","Iramba","Mkalama"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi", "Maharage","Uwele","Alizeti","Vitunguu","Maboga","Muhogo","Viazi vitamu","Mtama","Karanga","Njegere","Tumbaku","Ngano","Korosho"]
    },
 {
        "region": "Singida",
        "district": ["Iramba","Mkalama","Ikungi","Singida Rural","Manyoni","Itigi"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Uwele","Alizeti","Vitunguu","Maboga","Muhogo","Viazi vitamu","Mtama","Karanga","Njegere","Tumbaku","Ngano","Korosho","Pamba"]
    },
 {
        "region": "Singida",
        "district": ["Iramba","Mkalama","Ikungi","Singida Rural","Manyoni"],
        "ph_min": 7.4,
        "ph_max": 8.4,
        "acidity_level": "Slightly Alkaline",
        "recommended_crop": ["Mahindi"," Mpunga ","Maharage", "Uwele","Alizeti","Viazi vitamu","Mtama","Vitunguu","Maboga","Njegere","Ngano","Pamba"]
    },
 {
        "region": "Singida",
        "district": ["Mkalama"],
        "ph_min": 8.5,
        "ph_max": 14.0,
        "acidity_level": "Highly Alkaline",
        "recommended_crop": ["Mahindi"," Mpunga" ,"Maharage","Vitunguu","Pamba","Alizeti","Maboga"]
    },

{
        "region": "Songwe",
        "district": ["Ileje"],
        "ph_min": 0.0,
        "ph_max": 5.4,
        "acidity_level": "Highly Acidic",
        "recommended_crop": ["Mahindi","Maharage","Uwele","Alizeti","Muhogo","Viazi"]
    },
    {
        "region": "Songwe",
        "district": ["Momba,Ileje,Mbozi"],
        "ph_min": 5.5,
        "ph_max": 6.5,
        "acidity_level": "Moderately Acidic",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Migomba","Kahawa","Uwele","Ngano","Alizeti","Muhogo","Viazi"]
    },
    {
        "region": "Songwe",
        "district": ["Momba,Ileje,Mbozi"],
        "ph_min": 6.6,
        "ph_max": 7.3,
        "acidity_level": "Neutral",
        "recommended_crop": ["Mahindi","Mpunga","Maharage","Migomba","Kahawa","Chai","Hiliki","Uwele","Ngano","Alizeti","Muhogo","Viazi"]
    },
    # ➕ Add more entries here...
]

# Flatten entries: one row per district
rows = []
for entry in raw_data:
    for district in entry["district"]:
        rows.append({
            "region": entry["region"].strip(),
            "district": district.strip(),
            "ph_min": entry["ph_min"],
            "ph_max": entry["ph_max"],
            "acidity_level": entry["acidity_level"].strip(),
            "recommended_crop": [c.strip() for c in entry["recommended_crop"]]
        })

df = pd.DataFrame(rows)

# -------------------------------
# 2. Encode Target Labels
# -------------------------------
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['recommended_crop'])

# -------------------------------
# 3. Encode Input Features
# -------------------------------
X = df.drop(columns=['recommended_crop'])

region_encoder = LabelEncoder()
district_encoder = LabelEncoder()
acidity_encoder = LabelEncoder()

X['region'] = region_encoder.fit_transform(X['region'])
X['district'] = district_encoder.fit_transform(X['district'])
X['acidity_level'] = acidity_encoder.fit_transform(X['acidity_level'])

# -------------------------------
# 4. Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

print(" Model trained successfully.")

# -------------------------------
# 5. Save Everything
# -------------------------------
joblib.dump(model, 'crop_multi_model.pkl')
joblib.dump(mlb, 'crop_label_binarizer.pkl')
joblib.dump(region_encoder, 'region_encoder.pkl')
joblib.dump(district_encoder, 'district_encoder.pkl')
joblib.dump(acidity_encoder, 'acidity_encoder.pkl')
