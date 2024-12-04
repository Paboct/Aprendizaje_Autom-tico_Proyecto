# Aprendizaje_Automtico_Proyecto
 data_analysis -> En este archvio tenemos varias gráficas sobre las relaciones de algunas características.
                   Así como cierta información relativa a los datos, como el número de datos nulos, los percentiles...

 data_preprocessing -> Hecho

train_students_prep_minmax -> Hecho
 
train_students_prep_standard -> Hecho

decisive_model -> 'Acabado'

APS_Solver -> 'Acabado'

characteristics_selection.py -> Acabado

kfold -> Acabado

dt_metrics.csv -> Acabado

knn_metrics.csv -> Acabado

Prueba red neuronal

- 3 capas con 11 nodos
    - relu: 
    Neuronal Network accuracy:  [np.float64(88.95399201116511)]
    Neuronal Network f1:  [np.float64(88.90245642505784)]
    Neuronal Network precision:  [np.float64(88.99894942961589)]
    Neuronal Network recall:  [np.float64(88.95399201116511)]
    Neuronal Network error:  [np.float64(11.046007988834884)]
    Neuronal Network train:  0.891042909370791
    Neuronal Network test:  0.8895399201116512

    - tanh:
    Neuronal Network accuracy:  [88.98647673131525]
    Neuronal Network f1:  [88.94581358793175]
    Neuronal Network precision:  [89.00462180058902]
    Neuronal Network recall:  [88.98647673131525]
    Neuronal Network error:  [11.013523268684732]
    Neuronal Network train:  0.8951799114873966
    Neuronal Network test:  0.8898647673131526

    - logistic:
    Training loss did not improve more than tol=0.000010 for 10 consecutive epochs. Stopping.
    Neuronal Network accuracy:  [np.float64(88.36445449732902)]
    Neuronal Network f1:  [np.float64(88.28923195439617)]
    Neuronal Network precision:  [np.float64(88.46387476767313)]
    Neuronal Network recall:  [np.float64(88.36445449732902)]
    Neuronal Network error:  [np.float64(11.635545502670963)]
    Neuronal Network train:  0.8802193573215316
    Neuronal Network test:  0.8836445449732903

- 3 capas con 10 nodos
    - relu: 
    Neuronal Network accuracy:  [88.90466336204823]
    Neuronal Network f1:  [88.85123940949902]
    Neuronal Network precision:  [88.95332697871112]
    Neuronal Network recall:  [88.90466336204823]
    Neuronal Network error:  [11.09533663795178]
    Neuronal Network train:  0.8930633057533193
    Neuronal Network test:  0.8890466336204822
    
    - tanh:
    Neuronal Network accuracy:  [89.00332066028203]
    Neuronal Network f1:  [88.96703204059675]
    Neuronal Network precision:  [89.0119708378856]
    Neuronal Network recall:  [89.00332066028203]
    Neuronal Network error:  [10.996679339717986]
    Neuronal Network train:  0.8955166442178178
    Neuronal Network test:  0.8900332066028203
    
    - logistic:
    Neuronal Network accuracy:  [86.05683622888493]
    Neuronal Network f1:  [85.9173603108546]
    Neuronal Network precision:  [86.25693997650237]
    Neuronal Network recall:  [86.05683622888493]
    Neuronal Network error:  [13.943163771115064]
    Neuronal Network train:  0.859341928035405
    Neuronal Network test:  0.8605683622888494  

- 3 capas con 9 nodos
    - relu: 
    Neuronal Network accuracy:  [88.96000770008182]
    Neuronal Network f1:  [88.91988801266486]
    Neuronal Network precision:  [88.97495656823577]
    Neuronal Network recall:  [88.96000770008182]
    Neuronal Network error:  [11.03999229991819]
    Neuronal Network train:  0.8876274773908024
    Neuronal Network test:  0.8896000770008181
    
    - tanh:
    Neuronal Network accuracy:  [np.float64(88.94917946003176)]
    Neuronal Network f1:  [np.float64(88.90208757640549)]
    Neuronal Network precision:  [np.float64(88.98298288661618)]
    Neuronal Network recall:  [np.float64(88.94917946003176)]
    Neuronal Network error:  [np.float64(11.050820539968242)]
    Neuronal Network train:  0.8932076197806428
    Neuronal Network test:  0.8894917946003176
    
    - logistic:
    Neuronal Network accuracy:  [86.15669666490206]
    Neuronal Network f1:  [86.00004830633029]
    Neuronal Network precision:  [86.44103378681633]
    Neuronal Network recall:  [86.15669666490206]
    Neuronal Network error:  [13.843303335097934]
    Neuronal Network train:  0.859245718683856
    Neuronal Network test:  0.8615669666490205

- Con 3 capas, 10, 11, 9:
    Neuronal Network accuracy:  [89.11039992299919]
    Neuronal Network f1:  [89.07484092854784]
    Neuronal Network precision:  [89.11634996830695]
    Neuronal Network recall:  [89.11039992299919]
    Neuronal Network error:  [10.889600077000816]
    Neuronal Network train:  0.891956898210506
    Neuronal Network test:  0.891103999229992

- Con 2 capas, 10, 10
    Neuronal Network accuracy:  [89.03700851821552]
    Neuronal Network f1:  [89.00813942181888]
    Neuronal Network precision:  [89.03287110840171]
    Neuronal Network recall:  [89.03700851821552]
    Neuronal Network error:  [10.96299148178449]
    Neuronal Network train:  0.8897440831248797
    Neuronal Network test:  0.8903700851821551

- Con 3 capas, 100, 100, 50:
Training loss did not improve more than tol=0.000010 for 10 consecutive epochs. Stopping.
    Neuronal Network accuracy:  [np.float64(87.24072380769046)]
    Neuronal Network f1:  [np.float64(87.23797224333165)]
    Neuronal Network precision:  [np.float64(87.2356732288661)]
    Neuronal Network recall:  [np.float64(87.24072380769046)]
    Neuronal Network error:  [np.float64(12.759276192309542)]
    Neuronal Network train:  0.935876467192611
    Neuronal Network test:  0.8724072380769046

- Con 3 capas de 20:
    Neuronal Network accuracy:  [np.float64(89.21386977236632)]
    Neuronal Network f1:  [np.float64(89.20176854133676)]
    Neuronal Network precision:  [np.float64(89.2011430785718)]
    Neuronal Network recall:  [np.float64(89.21386977236632)]
    Neuronal Network error:  [np.float64(10.786130227633672)]
    Neuronal Network train:  0.8969116798152781
    Neuronal Network test:  0.8921386977236633

- Con 2 capas 20, 20:
    Neuronal Network accuracy:  [89.10869649990032]
    Neuronal Network f1:  [89.06915790921431]
    Neuronal Network precision:  [89.1258106982782]
    Neuronal Network recall:  [89.10869649990032]
    Neuronal Network error:  [10.891303500099689]
    Neuronal Network train:  0.8941042174107061
    Neuronal Network test:  0.8910869649990032

- Con 3 capas de 15:
    Neuronal Network accuracy:  [np.float64(88.9840704557486)]
    Neuronal Network f1:  [np.float64(88.94496550921535)]
    Neuronal Network precision:  [np.float64(88.99804956058291)]
    Neuronal Network recall:  [np.float64(88.9840704557486)]
    Neuronal Network error:  [np.float64(11.015929544251414)]
    Neuronal Network train:  0.8949393881085241
    Neuronal Network test:  0.8898407045574859

- Con 4 capas de 20
    Neuronal Network accuracy:  [89.12914429294759]
    Neuronal Network f1:  [89.09833278602679]
    Neuronal Network precision:  [89.13749573461098]
    Neuronal Network recall:  [89.12914429294759]
    Neuronal Network error:  [10.870855707052414]
    Neuronal Network train:  0.8978573280554979
    Neuronal Network test:  0.8912914429294758

- Con 3 capas de 30:
    Neuronal Network accuracy:  [np.float64(89.1079936474325)]
    Neuronal Network f1:  [np.float64(89.081414404818)]
    Neuronal Network precision:  [np.float64(89.10420317685673)]
    Neuronal Network recall:  [np.float64(89.1079936474325)]
    Neuronal Network error:  [np.float64(10.892006352567495)]
    Neuronal Network train:  0.9050894746969405
    Neuronal Network test:  0.8910799364743249

- Con 3 capas de 40:
    Neuronal Network accuracy:  [89.01341879941684]
    Neuronal Network f1:  [88.98139205284865]
    Neuronal Network precision:  [89.01847528367902]
    Neuronal Network recall:  [89.01341879941684]
    Neuronal Network error:  [10.98658120058315]
    Neuronal Network train:  0.9062306461385601
    Neuronal Network test:  0.8901341879941684

- Con 3 capas de 25:
    Neuronal Network accuracy:  [0.8922322242489615]
    Neuronal Network f1:  [0.8919248909661142]
    Neuronal Network precision:  [0.8923072696375229]
    Neuronal Network recall:  [0.8922322242489615]
    Neuronal Network error:  [0.10776777575103843]
    Neuronal Network train:  0.9011393545808686
    Neuronal Network test:  0.8922322242489615
- Con 3 capas de 25,25,24:
    Neuronal Network accuracy:  [89.30262425590904]
    Neuronal Network f1:  [89.27336750390413]
    Neuronal Network precision:  [89.30388279532174]
    Neuronal Network recall:  [89.30262425590904]
    Neuronal Network error:  [10.697375744090968]
    Neuronal Network train:  0.9002153239277051
    Neuronal Network test:  0.8930262425590904