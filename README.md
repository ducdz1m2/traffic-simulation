# Mô phỏng giao thông đô thị thông minh 

## Mục tiêu đề tài
Dự đoán mức độ nghiêm trọng và mô phỏng động sự lan truyền tắt nghẽn của tai nạn giao thông với ML và CA.

## Data
### Mô tả dữ liệu:
Sử dụng dataset 7.7 triệu mẫu về tai nạn giao thông trên toàn nước Mỹ (2016-2023) ([US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents))

Dữ liệu gồm 46 cột với mô tả như sau:
- ID: Mã định danh duy nhất cho mỗi vụ tai nạn.
- Source: Nguồn thu thập dữ liệu (tác giả thu thập từ 2 nguồn chính: MapQuest và Microsoft Bing Map Traffic [Mục Traffic Data Collection](https://arxiv.org/pdf/1906.05409)).
- **Severity: Mức độ nghiêm trọng của tai nạn.**
- Start_Time / End_Time: Thời điểm bắt đầu và kết thúc ảnh hưởng của vụ tai nạn đến giao thông.
- Start_Lat / Start_Lng: Vĩ độ và kinh độ tại điểm bắt đầu tai nạn.
- End_Lat / End_Lng: Vĩ độ và kinh độ tại điểm kết thúc (thường dùng để xác định đoạn đường bị ảnh hưởng).
- Distance(mi): Chiều dài đoạn đường (tính bằng dặm) bị ảnh hưởng bởi vụ tai nạn.
- Street, City, County, State, Zipcode: Các thông tin địa chỉ chi tiết.
- Temperature(F) / Wind_Chill(F): Nhiệt độ và nhiệt độ cảm nhận (độ F).
- Humidity(%): Độ ẩm.
- Pressure(in): Áp suất khí quyển.
- Visibility(mi): Tầm nhìn xa.
- Wind_Direction / Wind_Speed(mph): Hướng gió và tốc độ gió.
- Precipitation(in): Lượng mưa/tuyết.
- Weather_Condition: Tình trạng thời tiết cụ thể.
- Bump: Có gờ giảm tốc không?
- Crossing: Có vạch kẻ đường cho người đi bộ không?
- Junction: Có phải là nút giao lộ không?
- Railway: Có gần đường ray tàu hỏa không?
- Station: Có gần trạm dừng (xe buýt, tàu...) không?
- Traffic_Signal: Có đèn tín hiệu giao thông không?
- Amenity: Có gần các tiện ích (trạm xăng, nhà hàng...) không?
- Sunrise_Sunset: Cho biết lúc đó là ngày hay đêm.
- Civil / Nautical / Astronomical_Twilight: Các loại trạng thái hoàng hôn/bình minh khác nhau dựa trên vị trí mặt trời so với đường chân trời.
### Thống kê dữ liệu thiếu:

| Tên cột | Số lượng Null | Tỷ lệ phần trăm (%) |
| :--- | :---: | :---: |
| **End_Lat** | 3,402,762 | ~43.9% |
| **End_Lng** | 3,402,762 | ~43.9% |
| **Precipitation(in)** | 2,203,586 | ~28.4% |
| **Wind_Chill(F)** | 1,999,019 | ~25.8% |
| **Wind_Speed(mph)** | 571,233 | ~7.4% |
| **Visibility(mi)** | 177,098 | ~2.3% |
| **Wind_Direction** | 175,206 | ~2.3% |
| **Humidity(%)** | 174,144 | ~2.2% |
| **Weather_Condition** | 173,459 | ~2.2% |
| **Temperature(F)** | 163,853 | ~2.1% |
| **Pressure(in)** | 140,679 | ~1.8% |
| **Weather_Timestamp** | 120,228 | ~1.6% |
| **Airport_Code** | 22,635 | ~0.3% |
| **Timezone** | 7,808 | ~0.1% |

### Chi tiết về các cột quan trọng:
Tại [mô tả cột dữ liệu](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents), họ có đề cập về cột 'Severity': Shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short). Tức là mức độ nghiêm trọng tăng dần từ 1-4.