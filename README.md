# Linear Regression
1. Lý thuyết cơ sở
     - [trekhleb/homemade-machine-learning]
2. Code
   - Sử dụng thư viện
        ```python
            import numpy as np
            import pandas as pd
        ```
   - Hàm "Linear models" - lm()
     - 
     ```python
        def lm(Y,X):
            Xbar = np.append([np.ones(len(Y))], X, axis = 0)
            A = np.dot(Xbar,Xbar.T) 
            theta = np.dot(np.linalg.pinv(A),np.dot(Xbar,Y))
            return theta
     ```
     - Miêu tả - Mô hình Y ~ X có n quan sát và m biến độc lập
       - Gọi hàm: lm(Y,X)
       - Tham số: 
         - $Y = [y_1, y_2, ... , y_n]$ với yi là đầu ra của mô hình
         - $X =(X_1, X_2, ... , X_m)$ với $X_j = [x_1^j,x_2^j,..,x_n^j]$ các giá trị của thuộc tính thứ j
       - Công thức tìm theta : 
       $$ 
       \theta = (XX^T)^\dagger XY \\ 
       \text{với $A^\dagger$ là giả nghịch đảo của ma trận A} 
       $$

3. Bài toán
   - Đề: Tập dữ liệu **NhuCauXeBus.csv** cho biết dữ liệu về mức độ giao thông bằng xe bus ( Y - nghìn lượt khách/giờ), thu nhập bình quân đầu người ( $X_2$ - USD), dân số ( $X_3$ - nghìn người), mật độ dân số ( $X_4$ - người/dặm vuông) của 40 thành phố của Mỹ. Xét mô hình:
 $$Y = \theta_1 + \theta_2 X_2 + \theta_3 X_3 + \theta_4 X_4 + U$$ 
   - Code
   ```python
        data = pd.read_csv("NhuCauXeBus.csv")
        Y = data['Y'].values
        X2 = data['X2'].values
        X3 = data['X3'].values
        X4 = data['X4'].values
        theta = lm(Y,(X2,X3,X4))
        print(theta)
   ```
   - Kết quả: **[ 2.81570316e+03 -2.01273433e-01  1.57657508e+00  1.53421301e-01]**
   - So sách kết quả với hàm lm() có sẵn trên R studio cho gia kết quả như nhau !
    ![alt text][Rstudio]

[trekhleb/homemade-machine-learning]: https://github.com/trekhleb/homemade-machine-learning/tree/master/homemade/linear_regression
[Rstudio]:https://i.ibb.co/5M4WHqz/50999190-1267357760080992-7301813452559876096-n.png