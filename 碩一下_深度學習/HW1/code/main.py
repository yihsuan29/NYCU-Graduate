import os
from utils import * 
from model import *

args = parse_args()

result_dir = f"Case{args.case}/{args.a1}_h1{args.h1}_h2{args.h2}_lr{args.lr}_{args.opt}"
os.makedirs(result_dir, exist_ok=True)
config_path = os.path.join(result_dir, "config.txt")
result_path = os.path.join(result_dir, "result.txt")
result_img_path = os.path.join(result_dir, "result.png")
boundary_img_path = os.path.join(result_dir, "decision_boundary.png")
learning_curve_path = os.path.join(result_dir, "learning_curve.png")
curve_path = os.path.join(result_dir, "curve.txt")
with open(curve_path, "w") as f:
    pass
losses_path = os.path.join(result_dir, "loss.txt")

save_config(config_path, args)

if args.case== 1:    
    X,y = generate_linear(n=100)
else:
    X,y = generate_XOR_easy()

if args.a == 0:
    model = NN_wo_act(X.shape[1], args.h1, args.h2, 1)
else:
    model = NN(X.shape[1], args.h1, args.h2, 1, args.a1, args.a2, args.opt, args.lr)

losses = []
for epoch in range(args.epoch):
    y_pred = model.forward(X)
    loss = cross_entropy_loss(y, y_pred)
    if args.a==0:
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    dLoss = d_cross_entropy(y, y_pred)
    model.backward(dLoss, args.opt, args.lr)
    losses.append(loss)
    
    if (epoch+1) % 100 == 0:
        print(f"epoch {epoch+1}, loss: {loss:.4f}")
    # if epoch%100 ==0:
    #     evaluate(X,y,y_pred, X.shape[0], loss, curve_path)
    
evaluate(X,y,y_pred, X.shape[0], loss, result_path)
show_result(X,y,y_pred,result_img_path)
draw_boudary(model, X, y,boundary_img_path)
learning_curve(args.epoch, losses,learning_curve_path)

np.savetxt(losses_path, losses)