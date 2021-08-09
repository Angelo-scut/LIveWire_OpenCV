#include "lwocv.h"

cv::Mat calcImgGrad(const cv::Mat &img) {
    cv::Mat ret;
    cv::Mat gx, gy, ga;
    cv::Sobel(img, gx, CV_64F, 1,0);
    cv::Sobel(img, gy, CV_64F, 0,1);
    cv::cartToPolar(gx,gy, ret,ga);
    double vMax;
    cv::minMaxLoc(ret, NULL, &vMax, NULL,NULL);
    ret=1.0-ret/vMax;  // 因为都是返回cost，梯度越大，cost越小
    return ret;
}


cv::Mat calcCanny(const cv::Mat &img) {
    cv::Mat ret, tmp, imgb;
    img.copyTo(imgb);
    //    cv::blur(img, imgb, cv::Size(3,3));
    //    cv::adaptiveBilateralFilter(img,imgb,cv::Size(7,7),200);
    int param=9;
    cv::bilateralFilter(img,imgb,param,param*2,param/2.0);
    cv::Scalar vMean = cv::mean(imgb);
    std::cout << "mean = " << vMean << std::endl;
    //    tmp.reshape(0,1);
    //    cv::sort(tmp,tmp,CV_SORT_ASCENDING);
    //    double vMedian  = (double)tmp.at<uchar>(0,tmp.cols/2);
    double vMedian  = vMean[0];
    double sigma    = 0.3;
    double lower    = std::max(0.0, (1.0 - sigma) * vMedian);
    double upper    = std::min(255.0, (1.0 + sigma) * vMedian);
    std::cout << "lower=" << lower << ", upper=" << upper << std::endl;
    cv::Canny(imgb, ret, lower, upper, 3);
    ret.convertTo(ret,CV_64F);
    double vMax;
    cv::minMaxLoc(ret, NULL, &vMax, NULL,NULL);
    return (1.0 - ret/vMax);
}


cv::Mat calcLiveWireCostFcn(const cv::Mat &img) {
    cv::Mat imgg = img;
    if(img.channels()==3) {
        cv::cvtColor(img,imgg,CV_BGR2GRAY);
    }
    cv::Mat imgG = calcImgGrad(imgg);
    cv::Mat imgE = calcCanny(imgg);
    cv::Mat ret;
    double pG = 0.8;
    double pE = 0.2;
    //    std::cout << imgG.type() << "/" << imgE.type() << std::endl;
    ret = pG*imgG + pE*imgE;
    return ret;
}


cv::Mat normImage(const cv::Mat &img) {
    cv::Mat ret;
    cv::normalize(img, ret, 0,255, CV_MINMAX, CV_8U);
    return ret;
}


long fFindMinG(SEntry *pSList, long lLength)
{
    long    lMinPos = 0;
    float   flMin   = 1e15;
    SEntry  SE;
    for (long lI = 0; lI < lLength; lI++) {
        SE = *pSList++;
        if (SE.flG < flMin) {
            lMinPos = lI;
            flMin = SE.flG;
        }
    }
    return lMinPos;
}


long fFindLinInd(SEntry *pSList, long lLength, long lInd)
{
    SEntry SE;

    for (long lI = 0; lI < lLength; lI++) {
        SE = *pSList++;
        if (SE.lLinInd == lInd) return lI;
    }
    return -1; // If not found, return -1
}


void calcLiveWireP(const cv::Mat &imgS, int dX, int dY, cv::Mat &iPX, cv::Mat &iPY, double dRadius, int LISTMAXLENGTH)
{
    iPX.release();
    iPY.release();
    cv::Size siz = imgS.size();
    double *pdF	= (double*)(imgS.data);
    short   sNX = siz.width;
    short   sNY = siz.height;
    short   sXSeed = dX;
    short   sYSeed = dY;
    iPX = cv::Mat::zeros(siz, CV_8S);
    iPY = cv::Mat::zeros(siz, CV_8S);
    char *plPX    = (char*) (iPX.data);
    char *plPY    = (char*) (iPY.data);

    // Start of the real functionality
    long    lInd;
    long    lLinInd;
    long    lListInd = 0; // = length of list
    short   sXLowerLim;
    short   sXUpperLim;
    short   sYLowerLim;
    short   sYUpperLim;
    long    lNPixelsToProcess;
    long    lNPixelsProcessed = 0;

    float   flThisG;
    float   flWeight;

    SEntry  SQ, SR;

    cv::Mat lE = cv::Mat::zeros(siz, CV_8S);  // 用于记载某个点是否被处理过，准确来说是某个索引值
    char*   plE= (char*)(lE.data);
    SEntry *pSList = new SEntry[LISTMAXLENGTH];
    lNPixelsToProcess = ifMin(long(3.14*dRadius*dRadius + 0.5), long(sNX)*long(sNY)); // 设定了一个搜索范围，不然全图搜索太大了

    // Initialize active list with zero cost seed pixel.
    SQ.sX       = sXSeed;
    SQ.sY       = sYSeed;
    //    SQ.lLinInd  = ifLinInd(sXSeed, sYSeed, sNY);
    SQ.lLinInd  = ifLinInd(sXSeed, sYSeed, sNX); // 不知道原来是个指针索引index
    SQ.flG      = 0.0; // 储存cost
    pSList[lListInd++] = SQ;  // 用后加，也就是先pSList[0] = SQ, 然后lListInd++

    // While there are still objects in the active list and pixel limit not reached
    while ((lListInd) && (lNPixelsProcessed < lNPixelsToProcess)) { //这是以初始种子点为中心动态规划到以半径为dRadius的圆面积内各位置的最短路径，
		// 其中iPX和iPY储存最短路径的索引方向， 比如种子点到左上角，那左上角的iPX为1，iPY为1，也就是说从该点的右下角的位置过来的路径是最短的
        // ----------------------------------------------------------------
        // Determine pixel q in list with minimal cost and remove from
        // active list. Mark q as processed.
        lInd = fFindMinG(pSList, lListInd); // 找出pSList中cost值最小的位置
        SQ   = pSList[lInd];
        lListInd--;
        pSList[lInd] = pSList[lListInd]; // 将最后一个元素赋给pSList中cost值最小的位置， 可是这里只有点击一次才会调用
        plE[SQ.lLinInd]  = 1; // 表示这一个值已经处理过了
        // ----------------------------------------------------------------
        // Determine neighbourhood of q and loop over it 3*3的窗口
        sXLowerLim = ifMax(      0, SQ.sX - 1);
        sXUpperLim = ifMin(sNX - 1, SQ.sX + 1);
        sYLowerLim = ifMax(      0, SQ.sY - 1);
        sYUpperLim = ifMin(sNY - 1, SQ.sY + 1);
        for (short sX = sXLowerLim; sX <= sXUpperLim; sX++) {
            for (short sY = sYLowerLim; sY <= sYUpperLim; sY++) {
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                // Skip if pixel was already processed
                //                lLinInd = ifLinInd(sX, sY, sNY);
                lLinInd = ifLinInd(sX, sY, sNX);  // 只是简单地计算指针的索引值
                if (plE[lLinInd]) continue; // plE是一个与图像同样大小的标志位矩阵，如果某个像素位置置1了表明这一个位置已经被处理过了
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                // Compute the new accumulated cost to the neighbour pixel
                if ((abs(sX - SQ.sX) + abs(sY - SQ.sY)) == 1) flWeight = 0.71; else flWeight = 1; // 对角线的邻域权值更大一丢丢，具体就是sqrt(2)的问题
                flThisG = SQ.flG + float(pdF[lLinInd])*flWeight; // 原来pdf是cost矩阵，然后lLinInd是指针索引号
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                // Check whether r is already in active list and if the
                // current cost is lower than the previous
                lInd = fFindLinInd(pSList, lListInd, lLinInd); // 遍历pSList，如果iLinInd存在于pSList则返回1，否则返回-1
                if (lInd >= 0) {  // 如果在pSList了，则将它取出来更新（并没有丢掉）
                    SR = pSList[lInd];
                    if (flThisG < SR.flG) {
                        SR.flG = flThisG;
                        pSList[lInd] = SR;
                        plPX[lLinInd] = char(SQ.sX - sX);
                        plPY[lLinInd] = char(SQ.sY - sY); // 这里应该是记录某个像素位置的索引方向
                    }
                } else { // 否则加入
                    // - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // If r is not in the active list, add it!
                    SR.sX = sX;
                    SR.sY = sY;
                    SR.lLinInd = lLinInd;
                    SR.flG = flThisG;
                    pSList[lListInd++] = SR;
                    plPX[lLinInd] = char(SQ.sX - sX);
                    plPY[lLinInd] = char(SQ.sY - sY);
                    // - - - - - - - - - - - - - - - - - - - - - - - - - -
                }
            }
            // End of the neighbourhood loop.
            // ----------------------------------------------------------------
        }
        lNPixelsProcessed++;
    }
    // End of while loop
    delete pSList;
}


void calcLiveWireGetPath(const cv::Mat &ipx, const cv::Mat &ipy, cv::Point pxy, std::vector<cv::Point> &path, int iMAXPATH)
{
    path.clear();
    // Initialize the variables
    int iXS = pxy.x;
    int iYS = pxy.y;
    std::vector<int> iX(iMAXPATH,0);
    std::vector<int> iY(iMAXPATH,0);
    int iLength = 0;
    iX[iLength] = iXS;
    iY[iLength] = iYS;
    while ( (ipx.at<char>(iYS, iXS) != 0) || (ipy.at<char>(iYS, iXS) != 0) ) // We're not at the seed，如果ipx或者ipy有值表示该点具有离种子点最近的路径
    {
        iXS = iXS + ipx.at<char>(iYS, iXS);
        iYS = iYS + ipy.at<char>(iYS, iXS); // 开始逆推直到种子点，但是如果本身pxy就已经超出了parRadiusR，那么无法进入该循环
        iLength = iLength + 1;
        iX[iLength] = iXS;
        iY[iLength] = iYS;
    }
    for(int ii=iLength-2; ii>=0; ii--) { // 如果不超过两个点，那么直接返回空
        int tx = iX[ii];
        int ty = iY[ii];
        path.push_back(cv::Point(tx,ty));
    }
}


cv::Point calcIdealAnchor(const cv::Mat &imgS, cv::Point pxy, int rad) {  // 找出初始种子点以[-rad, rad]为邻域的区域灰度值最小值点
//    std::cout << "FUCK" << std::endl;
    int nc = imgS.cols;
    int nr = imgS.rows;
    int r0 = pxy.y;
    int c0 = pxy.x;
    //
    int rMin = r0-rad;
    int rMax = r0+rad;
    if(rMin<0) {
        rMin = 0;
    }
    if(rMax>=nr) {
        rMax = nr-1;
    }
    //
    int cMin = c0-rad;
    int cMax = c0+rad;
    if(cMin<0) {
        cMin = 0;
    }
    if(cMax>=nc) {
        cMax = nc-1;
    }
    //
    cv::Point ret(c0,r0);
    double valMin = imgS.at<double>(r0,c0);
    double tval;
    for(int rr=rMin; rr<rMax; rr++) {
        for(int cc=cMin; cc<cMax; cc++) {
            tval = imgS.at<double>(rr,cc);
            if(tval<valMin) {
                valMin = tval;
                ret.x = cc;
                ret.y = rr;
            }
        }
    }
    return ret;
}

///////////////////////////////////////////////////
OCVLiveWire::OCVLiveWire() {
    this->clean();
    initParams();
}

OCVLiveWire::OCVLiveWire(const cv::Mat &img, bool modeDebug) {
    initParams();
    setDebugMode(modeDebug);
    loadImage(img);
}

void OCVLiveWire::loadImage(const cv::Mat &img, bool modeDebug) {
    setDebugMode(modeDebug);
    imgf = calcLiveWireCostFcn(img);
    isStartPath = false;
    isLoaded    = true;
}

void OCVLiveWire::initParams() {
    isLoaded    = false;
    isStartPath = false;
    parRadiusA  = 4;
    parRadiusP  = 120;
    parRadiusPath   = 300;
}

bool OCVLiveWire::isLoadedData() const {
    return isLoaded;
}

bool OCVLiveWire::isStartedPath() const {
    return isStartPath;
}

void OCVLiveWire::setDebugMode(bool mode) {
    isDebug = mode;
}

cv::Point OCVLiveWire::getCPoint() const {
    return cPoint;
}

cv::Point OCVLiveWire::getMPoint() const {
    return mPoint;
}

void OCVLiveWire::clean() {
    this->isLoaded = false;
    imgf.release();
    iPX.release();
    iPY.release();
}

void OCVLiveWire::calcLWP(cv::Point p, bool isAnchorIdeal) {
    if(isLoaded) {
        cv::Point tp = p;
        if(isAnchorIdeal) {
            tp = calcIdealAnchor(imgf,p,parRadiusA);
        }
        calcLiveWireP(imgf, tp.x, tp.y, iPX, iPY, parRadiusP);
        isStartPath = true;
        cPoint = tp;
    }
}

void OCVLiveWire::calcLWPath(const cv::Point &p, bool isAnchorIdeal) {
    if(isLoaded && isStartPath) {
        cv::Point tp = p;
        if(isAnchorIdeal) {
            tp = calcIdealAnchor(imgf,p,parRadiusA);
        }
        double pDST = cv::norm(tp-cPoint); // cPoint是初始种子点
        bool isNoPath = true;
        if( (pDST>1.0) && (pDST<parRadiusPath)) { // 如果超出了局部动态规划范围，那么直接返回一个点，让他们连成直线，可是为啥和parRadiusR不一样？
            calcLiveWireGetPath(iPX,iPY, tp, path);
            if(path.size()>0) {
                isNoPath = false;
            }
        }
        if(isNoPath) {
            path.clear();
            path.push_back(tp);
        }
        this->mPoint = tp;
    }
}

void OCVLiveWire::incPath() {
    if(isLoaded) {
        if(path.size()>0) {
            pathTot.insert(pathTot.end(), path.begin(), path.end());
        }
    }
}

void OCVLiveWire::helpDrawImgPXY(const std::string &winName) {
    if(isLoaded) {
        cv::Mat tmp;
        cv::hconcat(iPX,iPY,tmp);
        cv::imshow(winName, normImage(tmp));
    }
}

void OCVLiveWire::helpDrawImgF(const std::string &winName) {
    if(isLoaded) {
        cv::imshow(winName, normImage(imgf));
    }
}
