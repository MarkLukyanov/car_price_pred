import streamlit as st
import pickle
import pandas as pd


def main():
    style = """<div style='background-color:pink; padding:12px'>
              <h1 style='color:black'>Предсказание стоимости автомобиля</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    left, right = st.columns((2,2))
    year = left.number_input('Какого года выпуска автомобиль?', step =1, value=2014)
    km_driven = right.number_input('Какой пробег у автомобиля?(в тыс.км)',  step=1, value=145500)
    mileage = left.number_input('Какой расход у автомобиля?(в kmpl)', step=0.01, format='%.2f', value=23.4)
    engine = right.number_input('Какой объём двигателя?(в см^3)', step=1, value=1248)
    max_power = left.number_input('Какая мощность двигателя?(в л.с.)', step=1, value=74)
    seats = right.number_input('Сколько сидений в автомобиле?', step=1, value=5)

    fuel = st.selectbox('Какой тип топлива у автомобиля?', ('Сжатый природный газ (метан)', 'Дизель',
                                                            'Сжиженный газ (пропан-бутан)', 'Бензин'))
    seller_type = st.selectbox('Какой тип продавца?', ('Дилер', 'Собственник', 'Дилер(Trustmark)'))
    transmission = st.selectbox('Какой тип КПП?', ('Механика', 'Автомат'))
    owner = st.selectbox('Какое количество владельцев было у автомобиля?', ('0', '1', '2', '3', '4 и более'))
    button = st.button('Рассчитать')
    # if button is pressed
    if button:
        # make prediction
        result = predict(year, km_driven, mileage, engine, max_power, seats, fuel, seller_type, transmission, owner)
        st.success(f'Цена автомобиля {result} у.е.')





# load the train model
with open('model.pickle', 'rb') as rf:
    model = pickle.load(rf)


def predict(year, km_driven, mileage, engine, max_power, seats, fuel, seller_type, transmission, owner):
    # processing user input
    f = [1, 0, 0, 0] if fuel == 'Сжатый природный газ (метан)' else [0, 1, 0, 0] if fuel == 'Дизель'\
        else [0, 0, 1, 0] if fuel == 'Сжиженный газ (пропан-бутан)' else [0, 0, 0, 1]

    s = [1, 0, 0] if seller_type == 'Дилер' else [0, 1, 0] if seller_type == 'Собственник' else [0, 0, 1]

    t = [1, 0] if transmission == 'Автомат' else [0, 1]

    o = [1, 0, 0, 0, 0] if owner == '1' else [0, 1, 0, 0, 0] if owner == '4 и более' else [0, 0, 1, 0, 0] if owner == '2'\
        else [0, 0, 0, 1, 0] if owner == '0' else [0, 0, 0, 0, 1]

    lists = [year, km_driven, mileage, engine, max_power, seats, *f, *s, *t, *o]
    df = pd.DataFrame(lists).transpose()
    # making predictions using the train model
    prediction = model.predict(df)
    result = int(prediction)
    return result



if __name__ == '__main__':
    main()