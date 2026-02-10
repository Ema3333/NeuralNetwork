using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
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

        const int numLayers = 4;
        float[][] layers = new float[numLayers][]; // store the fire value of each layer
        int[] numNeurons = new int[numLayers] { (width * height), 16, 16, 10 };

        float[][][] weight = new float[width * height][][];
        float[][] bias = new float[width * height][];

        float[] ExpVal = new float[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

        Random rnd = new Random();

        public MainWindow()
        {
            InitializeComponent();

            //random initialization of the weights and biases of the neural network

            for (int i = 0; i < numLayers - 1; i++)
            {
                weight[i] = new float[numNeurons[i]][];
                bias[i] = new float[numNeurons[i]];
                
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
                pixelMat[i] = new byte[width]; // x

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

        void NeuralNetwork(float[] ValArr)
        {
            // core structure of the neural network

            layers[0] = new float[width * height];

            for (int i = 0; i < width * height; i++)
            {
                layers[0][i] = ValArr[i];
            }

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
                    layers[i + 1][j] = 1f / (1f + (float)Math.Pow(Math.Exp(1), -(layers[i + 1][j])));
                }
            }

            for (int i = 0; i < numNeurons[numLayers - 1]; i++)
            {
                System.Diagnostics.Debug.WriteLine(layers[numLayers - 1][i]);
            }
        }

        void training()
        {
            float costVal = cost(layers, ExpVal);

            for (int i = numLayers; i < 0; i++)
            {
                for (int j = 0; j < numNeurons[i]; j++)
                {
                    for (int k = 0; k < numNeurons[i - 1]; k++)
                    {
                        if (i == numLayers)
                        {
                            
                        }
                    }
                }
            }
        }

        float gradient(float[][] Value)
        {
            for (int i = 0; i < numLayers - 1; i++)
            {
                weight[i] = new float[numNeurons[i]][];
                bias[i] = new float[numNeurons[i]];

                for (int j = 0; j < numNeurons[i + 1]; j++)
                {
                    

                    weight[i][j] = new float[numNeurons[i]];
                    for (int k = 0; k < numNeurons[i]; k++)
                    {
                        Value[i + 1] = 0;
                    }
                }
            }

            return 0;
        }

        float cost(float[][] Value, float[] ExpRes)
        {
            // cost function of the neural network

            float cost = 0;

            for (int i = 0; i < numNeurons[numLayers - 1]; i++)
            {
                cost += (float)Math.Pow(Value[numLayers - 1][i] - ExpRes[i], 2);
            }
            return cost;
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

            //End Debug

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    NumArr[count] = intMat[i][j];
                    count++;
                }
            }

            training();
            NeuralNetwork(NumArr);
        }

        private void ClearButton(object sender, RoutedEventArgs e)
        {
            this.Canvas.Strokes.Clear();
        }
    }
}
