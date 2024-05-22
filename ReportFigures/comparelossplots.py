import matplotlib.pyplot as plt
import json 

model_num = [11, 14]
loss_func = ''
history = {}

for i, model in enumerate(model_num):
    with open(rf'C:\Users\chloe\DE4\Masters\Models\Model_{model}_history.json', 'r') as f:
        history[model] = json.load(f)

    if history[model]['Loss Function'] == 'dice_mean_iou':
        loss_func = 'Dice / Mean IoU Loss'
    elif history[model]['Loss Function'] == 'dice_loss':
        loss_func = 'Dice Loss'
    elif history[model]['Loss Function'] == 'mean_iou_loss':
        loss_func = 'Mean IoU Loss'
    elif history[model]['Loss Function'] == 'dice_p_bce':
        loss_func = 'Dice / Weighted Binary Cross Entropy Loss'
    elif history[model]['Loss Function'] == 'weighted_bincrossentropy':
        loss_func = 'Weighted Binary Cross Entropy Loss'

    plt.plot(history[model_num[i]]['History']['val_loss'])

plt.title(f'Model {model_num[0]} and {model_num[1]} Validation Loss')
plt.ylabel(loss_func)
plt.xlabel('Epoch number')
plt.legend(['Model ' + str(model_num[0]), 'Model ' + str(model_num[1])], loc='upper left')
plt.show()
#plt.savefig(rf'C:\Users\chloe\DE4\Masters\Figures\Loss_{model_num[0]}_{model_num[1]}.pdf', dpi =300)
for i, model in enumerate(model_num):
    plt.plot(history[model_num[i]]['History']['val_binary_accuracy'])

plt.title(f'Model {model_num[0]} and {model_num[1]} Validation Binary Accuracy')
plt.ylabel('Binary Accuracy')
plt.xlabel('Epoch number')
plt.legend(['Model ' + str(model_num[0]), 'Model ' + str(model_num[1])], loc='upper left')
plt.show()
# plt.plot(history['History']['dice_coef'])
# plt.plot(history['History']['val_dice_coef'])
# plt.title(f'Model {model_num} Dice Coefficent')
# plt.ylabel('Dice Coefficent')
# plt.xlabel('Epoch number')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
# plt.savefig(rf'C:\Users\chloe\DE4\Masters\Figures\Dice_{model_num}.pdf', dpi =300)

# print("Max dice coef:", round(max(history['History']['dice_coef']),4))
# print("Max dice coef:", round(max(history['History']['val_dice_coef']),4))
# print("\nMin loss:", round(min(history['History']['loss']),4))
# print("Min val_loss:", round(min(history['History']['val_loss']),4))