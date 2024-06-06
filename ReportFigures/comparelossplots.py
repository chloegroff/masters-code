import matplotlib.pyplot as plt
import json 
import matplotlib

model_num = [24, 21, 23]
loss_func = ''
history = {}

matplotlib.rcParams.update({'font.size': 12, "font.family": "Times New Roman"})

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

plt.title(f'Test Dice/IoU Loss')
plt.ylabel(loss_func)
plt.xlabel('Epoch number')
plt.legend(['Training data: 94', 'Training data: 300', 'Training data: 500'], loc='upper left')
# plt.legend(['Model ' + str(model_num[0]), 'Model ' + str(model_num[1]), 'Model ' + str(model_num[2]), 'Model ' + str(model_num[3])], loc='upper left')

plt.savefig(rf'C:\Users\chloe\DE4\Masters\Figures\augmentation_tuning.pdf', dpi =300)
plt.show()

for i, model in enumerate(model_num):
    plt.plot(history[model_num[i]]['History']['val_binary_accuracy'])

plt.title(f'Model {model_num[0]} and {model_num[1]} Validation Binary Accuracy')
plt.ylabel('Binary Accuracy')
plt.xlabel('Epoch number')
plt.legend(['Training data: 94', 'Training data: 300', 'Training data: 500'], loc='upper left')
# plt.legend(['Model ' + str(model_num[0]), 'Model ' + str(model_num[1]), 'Model ' + str(model_num[2]), 'Model ' + str(model_num[3])], loc='upper left')
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