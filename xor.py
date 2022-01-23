from utils.all_utils import prepare_data,save_plot
import pandas as pd
from utils.model import Perceptron



def main(data,modelName,plotName,eta,epochs):
    df = pd.DataFrame(data)
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    model.save(filename=modelName,model_dir="model")
    save_plot(df,model,filename=plotName)

if __name__ == "__main__":
    XOR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,1,1,0]
    }
    ETA = 0.3  # 0 and 1
    EPOCHS = 10
    main(data=XOR,modelName = "xor.model",plotName="xor.png",eta=ETA,epochs=EPOCHS)


