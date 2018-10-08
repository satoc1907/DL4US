class test_and_makegraph :

    #make dataframe for graph
	def make_accdf(self,h5, type):
		model=load_model(h5)
		acc=[]
		loss_total=[]
		for i in  list(range(67,70))+list(range(71,76))+list(range(77,80)):
			validation_generator = test_datagen.flow_from_directory(os.path.join( 'your_directory',type+'_jpg','data_{}'.format(i)),target_size=(98, 98),batch_size=32,class_mode='binary')
			m=model.evaluate_generator(validation_generator)
			acc.append(m[1])
			loss_total.append(m[0])
			lst = [0.5 for i in range(11)]
			df=pd.DataFrame({'ACC': acc, 'loss_total':loss_total,'midline':lst},
                index=['67','68', '69', '71','72','73','74','75','77',
                      '78','79'])
        return(df)

    #make graph
    def plot_graph(df, title):
        s_bool = df['ACC'] >0.5
        s=s_bool.sum()/len(df['ACC'] )
        plt.ylim(0, 1.1)
        plt.xlabel('data_name')
        plt.ylabel('ACC')
        plt.title(title+'   '+'predictive value_{}'.format(s))
        plt.plot(df['midline'],color='orange')
        plt.bar(df.index,df['ACC'])

    
    #show resut
    def make_acc_list(h5, type):
        model=load_model(h5)
        n_oa=[]
        n_gbm=[]
        for i in  list(range(67,70))+list(range(71,76))+list(range(77,80)):
            validation_generator = test_datagen.flow_from_directory(os.path.join( '/Users/satoc/Desktop/Gr_add_test/',type+'_jpg','data_{}'.format(i)),target_size=(98, 98),batch_size=32,class_mode='binary')
            pd=model.predict_generator(validation_generator)
            n_oa.append(len(pd[pd>0.5]))
            n_gbm.append(len(pd[pd<0.5]))
        correcct=n_oa[0]+n_oa[1]+n_gbm[2]+n_oa[3]+n_gbm[4]+n_gbm[5]+n_gbm[6]+n_gbm[7]+n_gbm[8]+n_gbm[9]+n_gbm[10]
        total=sum(n_gbm)+sum(n_oa)
        q=correcct/total
        print(type)
        print('問題画像数',total)
        print('正解数',correcct)
        print('正解率',q)
        return(n_oa, n_gbm)
    
    # show VGG16_extract result
    model16 = VGG16(include_top=False, weights='imagenet')
    def make_acc_list_vgg16(h5, type):
    	model16 = VGG16(include_top=False, weights='imagenet')
        model=load_model(h5)
        n_oa=[]
        n_gbm=[]
        for i in  list(range(67,70))+list(range(71,76))+list(range(77,80)):
            validation_generator = test_datagen.flow_from_directory(os.path.join( '/Users/satoc/Desktop/Gr_add_test/',type+'_jpg','data_{}'.format(i)),target_size=(98, 98),batch_size=32,class_mode='binary')
            bottleneck_features_test = model16.predict_generator(validation_generator)
            pd=model.predict(bottleneck_features_test )
            n_oa.append(len(pd[pd>0.5]))
            n_gbm.append(len(pd[pd<0.5]))
        correcct=n_oa[0]+n_oa[1]+n_gbm[2]+n_oa[3]+n_gbm[4]+n_gbm[5]+n_gbm[6]+n_gbm[7]+n_gbm[8]+n_gbm[9]+n_gbm[10]
        total=sum(n_gbm)+sum(n_oa)
        q=correcct/total
        print(type)

    #以下model evalutaeがバグでつかえないので手計算を行う
        a_oa=np.array(n_oa)
        a_gbm=np.array(n_gbm)
        total=a_oa+a_gbm
        result=np.array([n_oa[0]/total[0], n_oa[1]/total[1], n_gbm[2]/total[2], n_oa[3]/total[3],
            n_gbm[4]/total[4], n_gbm[5]/total[5], n_gbm[6]/total[6],n_gbm[7]/total[7],
           n_gbm[8]/total[8],n_gbm[9]/total[9],n_gbm[10]/total[10],])
        TF=result>0.5
        TF=TF.tolist()
        TFc=TF.count(True)
        print('正解症例数',TFc)
        print('正解症例率',TFc/11)
        print('問題画像数',total)
        print('正解数',correcct)
        print('正解率',q)

        return(n_oa, n_gbm, TFc,result)


    #show RES50_extract result
    def make_acc_list_Res50(h5, type):
    	input_tensor = Input(shape=(294,294,3))
    	modelRes50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

    model=load_model(h5)
        n_oa=[]
        n_gbm=[]
        for i in  list(range(67,70))+list(range(71,76))+list(range(77,80)):
            validation_generator = test_datagen.flow_from_directory(os.path.join( '/Users/satoc/Desktop/Gr_add_test/',type+'_jpg','data_{}'.format(i)),target_size=(294, 294),batch_size=32,class_mode='binary')
            bottleneck_features_test = modelRes50.predict_generator(validation_generator)
            pd=model.predict(bottleneck_features_test )
            n_oa.append(len(pd[pd>0.5]))
            n_gbm.append(len(pd[pd<0.5]))
        correcct=n_oa[0]+n_oa[1]+n_gbm[2]+n_oa[3]+n_gbm[4]+n_gbm[5]+n_gbm[6]+n_gbm[7]+n_gbm[8]+n_gbm[9]+n_gbm[10]
        total=sum(n_gbm)+sum(n_oa)
        q=correcct/total
        print(type)
        #以下model evalutaeがバグでつかえないので手計算を行う
        a_oa=np.array(n_oa)
        a_gbm=np.array(n_gbm)
        total=a_oa+a_gbm
        result=np.array([n_oa[0]/total[0], n_oa[1]/total[1], n_gbm[2]/total[2], n_oa[3]/total[3],
            n_gbm[4]/total[4], n_gbm[5]/total[5], n_gbm[6]/total[6],n_gbm[7]/total[7],
           n_gbm[8]/total[8],n_gbm[9]/total[9],n_gbm[10]/total[10],])
        TF=result>0.5
        TF=TF.tolist()
        TFc=TF.count(True)
        print('正解症例数',TFc)
        print('正解症例率',TFc/11)
        print('問題画像数',total)
        print('正解数',correcct)
        print('正解率',q)
        
        return(n_oa, n_gbm, TFc)
