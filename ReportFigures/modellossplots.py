import matplotlib.pyplot as plt
import json 

model_num = 8
loss_func = 'Dice loss'
with open(rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}_history.json', 'r') as f:
     history = json.load(f)

if history['Loss Function'] == 'dice_mean_iou':
    loss_func = 'Dice / Mean IoU Loss'

plt.plot(history['History']['loss'])
plt.plot(history['History']['val_loss'])
plt.title(f'Model {model_num} Loss')
plt.ylabel(loss_func)
plt.xlabel('Epoch number')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig(rf'C:\Users\chloe\DE4\Masters\Figures\Loss_{model_num}.pdf', dpi =300)

plt.plot(history['History']['dice_coef'])
plt.plot(history['History']['val_dice_coef'])
plt.title(f'Model {model_num} Dice Coefficent')
plt.ylabel('Dice Coefficent')
plt.xlabel('Epoch number')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig(rf'C:\Users\chloe\DE4\Masters\Figures\Dice_{model_num}.pdf', dpi =300)

# print("Max acc:", round(max(history.history['acc']),4))
# print("Max val_acc:", round(max(history.history['val_acc']),4))
# print("\nMin loss:", round(min(history.history['loss']),4))
# print("Min val_loss:", round(min(history.history['val_loss']),4))