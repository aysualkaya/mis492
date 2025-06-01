#!/usr/bin/env python3
"""
🧪 Climate Utils Test Script
Bu script prepare_input_vector() fonksiyonunu farklı koordinatlarla test eder.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_model.climate_utils import prepare_input_vector, get_location_details
import time

# Test koordinatları - Farklı senaryolar
test_coordinates = [
    {
        "name": "🏙️ İstanbul Merkez",
        "lat": 41.0082,
        "lon": 28.9784,
        "month": 6,
        "expected": "Normal şehir merkezi - soil data mevcut olmalı"
    },
    {
        "name": "🌊 İstanbul Boğazı (Su üstü)", 
        "lat": 41.0205,
        "lon": 29.0044,
        "month": 6,
        "expected": "Fallback mekanizması devreye girmeli"
    },
    {
        "name": "🌾 Ankara Kırsal",
        "lat": 39.2734,
        "lon": 32.8597, 
        "month": 4,
        "expected": "Tarım alanı - detaylı soil data olabilir"
    },
    {
        "name": "🏖️ İzmir Kıyı",
        "lat": 38.4237,
        "lon": 27.1428,
        "month": 8,
        "expected": "Kıyı bölgesi - tuzlu toprak olasılığı"
    },
    {
        "name": "🏔️ Hatay Sınır",
        "lat": 36.2048,
        "lon": 36.1611,
        "month": 10, 
        "expected": "Sınır bölgesi - veri eksikliği olabilir"
    },
    {
        "name": "🌿 Bursa Tarım",
        "lat": 40.1826,
        "lon": 29.0665,
        "month": 5,
        "expected": "Verimli tarım toprakları"
    }
]

def test_coordinate(test_case):
    """Tek bir koordinat için test yapar"""
    print(f"\n{'='*60}")
    print(f"📍 TEST: {test_case['name']}")
    print(f"📍 Koordinat: ({test_case['lat']}, {test_case['lon']})")
    print(f"📅 Ay: {test_case['month']}")
    print(f"💭 Beklenen: {test_case['expected']}")
    print(f"{'='*60}")
    
    try:
        # Lokasyon detayı al
        location = get_location_details(test_case['lat'], test_case['lon'])
        print(f"🌍 Lokasyon: {location}")
        
        # Input vector hesapla
        start_time = time.time()
        input_vector = prepare_input_vector(
            test_case['lat'], 
            test_case['lon'], 
            test_case['month']
        )
        end_time = time.time()
        
        # Sonuçları göster
        print(f"\n✅ BAŞARILI!")
        print(f"⏱️ Süre: {end_time - start_time:.2f} saniye")
        print(f"📊 Input Vector: {input_vector}")
        print(f"📏 Vector Uzunluğu: {len(input_vector)}")
        
        # Vector elemanlarını açıkla
        if len(input_vector) == 7:
            labels = ["Soil Type (encoded)", "pH", "K", "P", "N", "Temperature (°C)", "Humidity (%)"]
            print(f"\n📋 Detaylı Analiz:")
            for i, (label, value) in enumerate(zip(labels, input_vector)):
                print(f"   {i}: {label} = {value}")
        
        return True, input_vector
        
    except Exception as e:
        print(f"\n❌ HATA OLUŞTU!")
        print(f"🚨 Hata Mesajı: {str(e)}")
        print(f"📄 Hata Tipi: {type(e).__name__}")
        return False, None

def main():
    """Ana test fonksiyonu"""
    print("🧪 CLIMATE UTILS TEST BAŞLIYOR")
    print("🔬 prepare_input_vector() fonksiyonu test ediliyor...")
    
    successful_tests = 0
    total_tests = len(test_coordinates)
    results = []
    
    for test_case in test_coordinates:
        success, vector = test_coordinate(test_case)
        results.append({
            'name': test_case['name'],
            'success': success,
            'vector': vector
        })
        
        if success:
            successful_tests += 1
        
        # Test aralarında kısa bekleme
        time.sleep(1)
    
    # Özet rapor
    print(f"\n{'='*60}")
    print(f"📊 TEST ÖZETI")
    print(f"{'='*60}")
    print(f"✅ Başarılı: {successful_tests}/{total_tests}")
    print(f"❌ Başarısız: {total_tests - successful_tests}/{total_tests}")
    print(f"📈 Başarı Oranı: {(successful_tests/total_tests)*100:.1f}%")
    
    # Başarılı testlerin vector'larını karşılaştır
    successful_vectors = [r['vector'] for r in results if r['success']]
    if len(successful_vectors) > 1:
        print(f"\n🔍 VECTOR ANALİZİ:")
        print(f"📏 Tüm vector'lar 7 elemanlı mı: {all(len(v) == 7 for v in successful_vectors)}")
        
        # Min/max değerleri
        if all(len(v) == 7 for v in successful_vectors):
            for i in range(7):
                values = [v[i] for v in successful_vectors]
                print(f"   Element {i}: Min={min(values):.2f}, Max={max(values):.2f}")
    
    print(f"\n🏁 TEST TAMAMLANDI!")
    
    if successful_tests == total_tests:
        print("🎉 Tüm testler başarılı! Sistem tamamen çalışıyor.")
    elif successful_tests > 0:
        print("⚠️ Bazı testler başarısız oldu. Detayları yukarıda kontrol edin.")
    else:
        print("🚨 Hiçbir test başarılı olmadı. Sistem konfigürasyonunu kontrol edin!")

if __name__ == "__main__":
    main()