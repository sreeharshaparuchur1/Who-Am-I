import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
	


class Skin_Detect():
	def __init__(self):
	#Constractor that does nothing
		pass
	#RGB bounding rule
	def Rule_A(self,BGR_Frame,plot=False):
		B_Frame, G_Frame, R_Frame =  [BGR_Frame[...,BGR] for BGR in range(3)]# [...] is the same as [:,:]
		BRG_Max = np.maximum.reduce([B_Frame, G_Frame, R_Frame])
		BRG_Min = np.minimum.reduce([B_Frame, G_Frame, R_Frame])
		#at uniform daylight, The skin colour illumination's rule is defined by the following equation :
		Rule_1 = np.logical_and.reduce([R_Frame > 95, G_Frame > 40, B_Frame > 20 ,
	                                 BRG_Max - BRG_Min > 15,abs(R_Frame - G_Frame) > 15, 
	                                 R_Frame > G_Frame, R_Frame > B_Frame])
		#the skin colour under flashlight or daylight lateral illumination rule is defined by the following equation :
		Rule_2 = np.logical_and.reduce([R_Frame > 220, G_Frame > 210, B_Frame > 170,
	                         abs(R_Frame - G_Frame) <= 15, R_Frame > B_Frame, G_Frame > B_Frame])
		#Rule_1 U Rule_2
		RGB_Rule = np.logical_or(Rule_1, Rule_2)
		if plot == True:
			#original image RGB color
			rgb_img = cv2.merge([R_Frame,G_Frame,B_Frame])# combine the RGB components to get the original image
			fig = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			fig.suptitle('RGB space', fontsize=16)
			cmaps = [None, 'gray']
			titles = ['Original-Image','RGB-Mask']

			#black and white image
			img_bw = RGB_Rule.astype(np.uint8)*255  #Test the output
			images = [rgb_img,img_bw]

			for i in range(2):
				ax = fig.add_subplot(1, 2, i+1)
				ax.axes.get_xaxis().set_visible(False)
				ax.axes.get_yaxis().set_visible(False)
				ax.set_title(titles[i])
				ax.imshow(images[i],cmap=cmaps[i])

		#return the RGB mask
		return RGB_Rule
	def lines(self,axis):
		'''return a list of lines for a give axis'''
		line = []
		lvals = [1.5862,0.3448,-1.005,-1.15,-2.2857]
		rvals = [20,76.2069,234.5652,301.75,432.85]
		for i in range(5):
			line.append(lvals[i]*axis+rvals[i])
		return line

	def Rule_B(self,YCrCb_Frame,plot=False):
		Y_Frame,Cr_Frame, Cb_Frame = [YCrCb_Frame[...,YCrCb] for YCrCb in range(3)]
		line1,line2,line3,line4,line5 = self.lines(Cb_Frame)
		YCrCb_Rule = np.logical_and.reduce([line1 - Cr_Frame >= 0,
											line2 - Cr_Frame <= 0,
											line3 - Cr_Frame <= 0,
											line4 - Cr_Frame >= 0,
											line5 - Cr_Frame >= 0])
		# Create a plot
		if plot == True:
			fig1 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			ax1 = fig1.add_subplot(1, 1, 1)
			ax1.scatter(Cb_Frame, Cr_Frame, alpha=0.8, c='black', edgecolors='none', s=10, label="Cr")
			ax1.set_xlim([0, 255])
			ax1.set_ylim([0, 255])
			ax1.set_xlabel('Cb')
			ax1.set_ylabel('Cr')
			ax1.xaxis.set_label_coords(0.5, -0.025)

			#draw a line
			x_axis = np.linspace(0, 255,100)
			line1,line2,line3,line4,line5 = self.lines(x_axis)
			lines = [line1,line2,line3,line4,line5]
			colors = ['b','g','r','m','y']
			labels = ['line1','line3','line3','line4','line5']
			for i in range(5):
				ax1.plot(x_axis, lines[i], alpha=0.5, c=colors[i], label=labels[i])	
			plt.title('Bounding Rule for Cb-Cr space')
			plt.legend(loc=(1,0.7))

			#plot the Y Cr and Cb components on a different figure
			fig2 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			fig2.suptitle('Y-Cr-Cb components', fontsize=16)
			
			distribution = ['Y','Cb','Cr']
			ravels = [Y_Frame,Cb_Frame,Cr_Frame]
			for i in range(3):
				ax = fig2.add_subplot(3, 1, i+1)
				ax.set_title(distribution[i])
				ax.title.set_position([0.9, 0.95])
				ax.set_xlabel('pixel intensity')
				ax.xaxis.set_label_coords(0.5, -0.025)
				ax.set_ylabel('number of pixels')
				ax.hist((ravels[i]).ravel(), bins=256, range=(0, 256), fc='b', ec='b')

			img_bw = (YCrCb_Rule.astype(np.uint8))*255  
			fig3 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			ax1 = fig3.add_subplot(1, 1, 1)
			ax1.axes.get_xaxis().set_visible(False)
			ax1.axes.get_yaxis().set_visible(False)
			ax1.set_title('CrCb-Mask')
			#plot as a Grayscale image
			ax1.imshow(img_bw,cmap='gray', vmin=0, vmax=255,interpolation='nearest')
		return YCrCb_Rule

	def Rule_C(self,HSV_Frame,plot=False):
		Hue,Sat,Val = [HSV_Frame[...,i] for i in range(3)]
		#i changed the value of the paper 50 instead of 25 and 150 instead of 230 based on my plots
		HSV_ = np.logical_or(Hue < 50, Hue > 150)
		if plot == True:
			#Plot Hue(x_axis) vs Value(y_axis)
			fig1 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			func = [Val,Sat]
			ylabels = ['Val','Sat']
			titles = ['HSV skin color Distribution H vs V','HSV skin color Distribution H vs S']
			for i in range(2):
				ax = fig1.add_subplot(1, 2, i+1)
				ax.scatter(Hue,func[i],alpha=0.8, c='b', edgecolors='none', s=10, label="Cr")
				ax.set_xlim([0, 255])
				ax.set_ylim([0, 255])
				ax.set_xlabel('Hue')
				ax.set_ylabel(ylabels[i])
				ax.set_title(titles[i])

			#plot Hue mask
			Hue_bw = HSV_.astype(np.uint8)*255  #Test the output
			fig2 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			ax1 = fig2.add_subplot(1, 1, 1)
			ax1.axes.get_xaxis().set_visible(False)
			ax1.axes.get_yaxis().set_visible(False)
			ax1.set_title('Hue-Mask')
			#plot as a Grayscale image
			ax1.imshow(Hue_bw,cmap='gray', vmin=0, vmax=255,interpolation='nearest')
		return HSV_

	def RGB_H_CbCr(self, Frame_,plot=False):
		Ycbcr_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2YCrCb)
		HSV_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2HSV)
		#Rule A ∩ Rule B ∩ Rule C
		skin_ = np.logical_and.reduce([self.Rule_A(Frame_), self.Rule_B(Ycbcr_Frame), self.Rule_C(HSV_Frame)])
		if plot == True:
			skin_bw= skin_.astype(np.uint8) 
			skin_bw*=255
			#RGB original image
			RGB_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2RGB)
			seg = cv2.bitwise_and(Frame_,Frame_,mask=skin_bw)
			#plot as a Grayscale image
			cv2.imshow("Extracted Skin",seg)
		return np.asarray(skin_, dtype=np.uint8)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("please give me an image !!!")
		sys.exit(0)
	image = sys.argv[1]
	try:
		img = np.array(cv2.imread(image), dtype=np.uint8)
	except:
		print('Error while loading the Image,image does not exist!!!!')
		sys.exit(1)
	test = Skin_Detect()
	YCrCb_Frames = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	HSV_Frames = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	test.Rule_A(img,True)
	test.Rule_B(YCrCb_Frames,True)
	test.Rule_C(HSV_Frames,True)
	test.RGB_H_CbCr(img,True)
	plt.show()
	cv2.waitKey(0)