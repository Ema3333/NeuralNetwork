using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WpfApp1
{
    public partial class MainWindow : Window
    {
        // define the width and height to which the canvas will be shrinken down

        const int width = 28;
        const int height = 28;

        const int epoch = 50;

        const float learningRate = 0.01f;

        const int numLayers = 8;
        float[][] layers = new float[numLayers][]; // store the fire value of each layer
        int[] numNeurons = new int[numLayers] { (width * height), 600, 500, 400, 300, 200, 100, 10 };

        float[][][] weight = new float[numLayers][][];
        float[][] bias = new float[numLayers][];

        Random rnd = new Random();

        public MainWindow()
        {
            InitializeComponent();

            //random initialization of the weights and biases of the neural network

            for (int i = 0; i < numLayers - 1; i++)
            {
                weight[i] = new float[numNeurons[i + 1]][];
                bias[i] = new float[numNeurons[i + 1]];

                for (int j = 0; j < numNeurons[i + 1]; j++)
                {
                    bias[i][j] = (rnd.Next(1, 100) - 50f) / 100f;

                    weight[i][j] = new float[numNeurons[i]];
                    for (int k = 0; k < numNeurons[i]; k++)
                    {
                        weight[i][j][k] = (rnd.Next(1, 100) - 50f) / 100f;
                    }
                }
            }
        }
        private float[][] canvasToMatrix()
        {
            // render as bitmap the canvas than covert the bitmap render to a matrix of 0 and 1 where 0 is white (neuron off) and 1 is black (neuron on)

            double dpi = (28d * 96d) / Canvas.ActualWidth;

            int x = 0;
            int y = 0;

            int count = 0;

            byte[] pixel = new byte[width * height * 4];
            byte[][] pixelMat = new byte[width][];
            float[][] intMat = new float[width][];

            RenderTargetBitmap rtb = new RenderTargetBitmap(width, height, dpi, dpi, PixelFormats.Default);

            rtb.Render(Canvas);
            int stride = ((width * rtb.Format.BitsPerPixel + 31) / 32) * 4;

            rtb.CopyPixels(pixel, stride, 0);

            for (int i = 0; i < height; i++)
            {
                pixelMat[i] = new byte[width];

                for (int j = 0; j < width; j++)
                {
                    int index = (i * stride) + (j * 4);
                    pixelMat[i][j] = pixel[index];
                }
            }

            for (int i = 0; i < height; i++)
            {
                intMat[i] = new float[width];
                for (int j = 0; j < width; j++)
                {
                    if (pixelMat[i][j] == 0)
                    {
                        intMat[i][j] = 1;
                    }
                    else
                    {
                        intMat[i][j] = 0;
                    }
                }
            }

            return intMat;
        }

        void NeuralNetwork(float[] ValArr, bool print)
        {
            // core structure of the neural network

            layers[0] = new float[width * height];

            for (int i = 0; i < width * height; i++)
            {
                layers[0][i] = ValArr[i];
            }

            // compute first value of each neuron
            for (int i = 0; i < numLayers - 1; i++)
            {
                layers[i + 1] = new float[numNeurons[i + 1]];
                for (int j = 0; j < numNeurons[i + 1]; j++)
                {
                    for (int k = 0; k < numNeurons[i]; k++)
                    {
                        layers[i + 1][j] += layers[i][k] * weight[i][j][k];
                    }
                    layers[i + 1][j] += bias[i][j];
                    layers[i + 1][j] = 1f / (1f + (float)Math.Pow(Math.Exp(1), -(layers[i + 1][j]))); // sigmoide
                }
            }

            if (print)
            {
                float num = -1;
                float tempOld = -1;
                float temp = 0;

                for (int i = 0; i < numNeurons[numLayers - 1]; i++)
                {
                    temp = layers[numLayers - 1][i];

                    if (temp > tempOld)
                    {
                        num = i;
                        tempOld = temp;
                    }

                    System.Diagnostics.Debug.WriteLine(layers[numLayers - 1][i]);
                }

                if (num == -1)
                {
                    System.Diagnostics.Debug.WriteLine("error");
                    NumGuess.Content = "error";
                }

                System.Diagnostics.Debug.WriteLine(num);
                NumGuess.Content = num;
            }
        }

        (float[][][], float[][]) gradient(float[] expVal)
        {
            // compute the gradient

            float[][] biasGradient = new float[numLayers][];
            float[][][] weightGradient = new float[numLayers][][];
            float[][] delta = new float[numLayers][];
            float sum = 0;

            // inizialization
            for (int i = 0; i <= numLayers - 1; i++)
            {
                biasGradient[i] = new float[numNeurons[i]];
                delta[i] = new float[numNeurons[i]];
            }

            // compute delta last layer
            for (int i = 0; i < numNeurons[numNeurons.Length - 1]; i++)
            {
                delta[numLayers - 1][i] = (layers[numLayers - 1][i] - expVal[i]) * layers[numLayers - 1][i] * (1 - layers[numLayers - 1][i]);
                //System.Diagnostics.Debug.WriteLine(delta[numLayers - 1][i]);
            }

            // compute delta hidden layer (backpropagation)
            for (int i = numLayers - 1; i > 0; i--)
            {
                for (int j = 0; j < numNeurons[i - 1]; j++)
                {
                    for (int k = 0; k < numNeurons[i]; k++)
                    {
                        sum += delta[i][k] * weight[i - 1][k][j];
                    }

                    delta[i - 1][j] = sum * layers[i - 1][j] * (1 - layers[i - 1][j]);
                    sum = 0;
                }
            }

            //compute gradient
            for (int i = 0; i < numLayers - 1; i++)
            {
                weightGradient[i] = new float[numNeurons[i + 1]][];
                for (int j = 0; j < numNeurons[i + 1]; j++)
                {
                    weightGradient[i][j] = new float[numNeurons[i]];

                    biasGradient[i][j] = delta[i + 1][j];
                    for (int k = 0; k < numNeurons[i]; k++)
                    {
                        weightGradient[i][j][k] = delta[i + 1][j] * layers[i][k];
                    }
                }
            }

            return (weightGradient, biasGradient);
        }

        void training()
        {
            float[] ExpVal = new float[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            float loss = 0f;

            string pathImg = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "train-images.idx3-ubyte");
            string pathLab = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "train-labels.idx1-ubyte");

            byte[] fileImg = File.ReadAllBytes(pathImg);
            byte[] fileLab = File.ReadAllBytes(pathLab);
            byte[] tempImg = new byte[4];   
            byte[] tempLab = new byte[4];

            for (int i = 0; i < 4; i++)
            {
                tempImg = new byte[4];
                tempLab = new byte[4];

                for (int j = 0; j < 4; j++)
                {
                    tempImg[j] = fileImg[j + (4 * i)];
                    tempLab[j] = fileLab[j + (4 * i)];
                }

                Array.Reverse(tempImg);
                Array.Reverse(tempLab);

                for (int j = 0; j < 4; j++)
                {
                    fileImg[j + (4 * i)] = tempImg[j];
                    fileLab[j + (4 * i)] = tempLab[j];
                }
            }

            int nImg = BitConverter.ToInt32(fileImg, 4);
            int row = BitConverter.ToInt32(fileImg, 8);
            int columns = BitConverter.ToInt32(fileImg, 12);
            int size = row * columns;

            for (int q = 0; q < epoch; q++)
            {
                for (int f = 0; f < nImg; f++)
                {
                    loss = 0f;
                    ExpVal = new float[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                    float[] trainingVal = new float[size];

                    for (int r = 0; r < size; r++)
                    {
                        trainingVal[r] = fileImg[((f * size) + 16) + r] / 255f;
                    }

                    NeuralNetwork(trainingVal, false);

                    int index = fileLab[f + 8];
                    ExpVal[index] = 1;

                    for (int p = 0; p < 10; p++)
                    {
                        loss += (float)Math.Pow(ExpVal[p] - layers[numLayers - 1][p], 2);
                    }

                    if (f % 100 == 0)
                    {
                        System.Diagnostics.Debug.WriteLine("perdita: " + loss);
                    }

                    (var wG, var bG) = gradient(ExpVal);

                    for (int i = 0; i < numLayers - 1; i++)
                    {
                        for (int j = 0; j < numNeurons[i + 1]; j++)
                        {
                            bias[i][j] = bias[i][j] - (bG[i][j] * learningRate);

                            for (int k = 0; k < numNeurons[i]; k++)
                            {
                                weight[i][j][k] = weight[i][j][k] - (wG[i][j][k] * learningRate);
                            }
                        }
                    }
                }

                save();
            }
        }

        void save()
        {
            string pathData = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "trainData.dat");

            using (BinaryWriter write = new BinaryWriter(File.Open(pathData, FileMode.Create)))
            {
                for (int i = 0; i < numLayers - 1; i++)
                {
                    for (int j = 0; j < numNeurons[i + 1]; j++)
                    {
                        write.Write(bias[i][j]);

                        for (int k = 0; k < numNeurons[i]; k++)
                        {
                            write.Write(weight[i][j][k]);
                        }
                    }
                }
            }
        }

        void load()
        {
            string pathData = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "trainData.dat");

            using (BinaryReader read = new BinaryReader(File.Open(pathData, FileMode.Open)))
            {
                for (int i = 0; i < numLayers - 1; i++)
                {
                    for (int j = 0; j < numNeurons[i + 1]; j++)
                    {
                        bias[i][j] = read.ReadSingle();

                        for (int k = 0; k < numNeurons[i]; k++)
                        {
                            weight[i][j][k] = read.ReadSingle();
                        }
                    }
                }
            }

            System.Diagnostics.Debug.WriteLine("File loaded successfully");
        }

        private void IdentifyButton(object sender, RoutedEventArgs e)
        {
            // call canvasToMatrix to get the matrix of the canvas and then call the neural network function to get the output

            float[][] intMat = canvasToMatrix();
            float[] NumArr = new float[width * height];

            int count = 0;

            //Debug

            for (int i = 0; i < height; i++)
            {
                string riga = "";
                for (int j = 0; j < width; j++)
                {
                    riga += intMat[i][j] + " ";
                }
                System.Diagnostics.Debug.WriteLine(riga);
            }

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    NumArr[count] = intMat[i][j];
                    count++;
                }
            }

            NeuralNetwork(NumArr, true);
        }

        private void ClearButton(object sender, RoutedEventArgs e)
        {
            this.Canvas.Strokes.Clear();
        }

        private void TrainButton(object sender, RoutedEventArgs e)
        {
            training();
        }

        private void LoadData(object sender, RoutedEventArgs e)
        {
            load();
        }
    }
}
