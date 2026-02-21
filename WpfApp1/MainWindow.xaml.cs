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

        const int numLayers = 5;
        float[][] layers = new float[numLayers][]; // store the fire value of each layer
        int[] numNeurons = new int[numLayers] { (width * height), 512, 256, 128, 47 };

        float[][][] weight = new float[numLayers][][];
        float[][] bias = new float[numLayers][];

        bool stop = false;
        bool stopAccuracy = false;

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

            loadAsync();
        }
        private float[][] canvasToMatrix()
        {
            // render as bitmap the canvas than covert the bitmap render to a matrix of 0 and 1 where 0 is white (neuron off) and 1 is black (neuron on)

            double dpi = (28d * 96d) / Canvas.ActualWidth;

            byte[] pixel = new byte[width * height * 4];
            byte[][] pixelMat = new byte[width][];
            float[][] intMat = new float[width][];
            bool temp = true;
            int firstI = 0;
            int firstJ = 0;
            int lastI = 0;
            int lastJ = 0;

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

            // center the number
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (intMat[i][j] == 1 && temp)
                    {
                        firstI = i;
                        firstJ = j;
                        temp = false;
                    }

                    if (intMat[i][j] == 1)
                    {
                        lastI = i;
                        lastJ = j;
                    }
                }
            }

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    
                }
            }

            return intMat;
        }

        float NeuralNetwork(float[] ValArr, bool print)
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

            float num = -1;
            float tempOld = -1;
            float temp = 0;
            char outputChar;

            for (int i = 0; i < numNeurons[numLayers - 1]; i++)
            {
                temp = layers[numLayers - 1][i];

                if (temp > tempOld)
                {
                    num = i;
                    tempOld = temp;
                }

                if (print)
                {
                    System.Diagnostics.Debug.WriteLine(layers[numLayers - 1][i] + ": " + i);
                }
            }

            if (print)
            {
                if (num <= 9 && num >= 0)
                {
                    System.Diagnostics.Debug.WriteLine(num);
                    NumGuess.Content = num.ToString();
                }
                else
                {
                    outputChar = (char)(num + 55);
                }
            }


            return num;
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
            float[] ExpVal = new float[47];

            for (int i = 0; i < ExpVal.Length; i++)
            {
                ExpVal[i] = 0;
            }

            float loss = 0f;

            string pathNumImg = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "emnist-digits-train-images-idx3-ubyte");
            string pathNumLab = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "emnist-digits-train-labels-idx1-ubyte");
            string pathCharImg = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "emnist-letters-train-images-idx3-ubyte");
            string pathCharLab = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "emnist-letters-train-labels-idx1-ubyte");

            byte[] fileNumImg = LittleEndianConv(File.ReadAllBytes(pathNumImg), 4);
            byte[] fileNumLab = LittleEndianConv(File.ReadAllBytes(pathNumLab), 2);
            byte[] fileCharImg = LittleEndianConv(File.ReadAllBytes(pathCharImg), 4);
            byte[] fileCharLab = LittleEndianConv(File.ReadAllBytes(pathCharLab), 2);

            byte[] cleanNumImg = removeMetadata(fileNumImg, 16);
            byte[] cleanNumLab = removeMetadata(fileNumLab, 8);
            byte[] cleanCharImg = removeMetadata(fileCharImg, 16);
            byte[] cleanCharLab = removeMetadata(fileCharLab, 8);

            for (int i = 0; i < cleanCharLab.Length; i++)
            {
                cleanCharLab[i] += 9;
            }

            byte[] fileImg = mergeFileContent(cleanNumImg, cleanCharImg);
            byte[] fileLab = mergeFileContent(cleanNumLab, cleanCharLab);

            int nImg = BitConverter.ToInt32(fileNumImg, 4) + BitConverter.ToInt32(fileCharImg, 4);
            int row = BitConverter.ToInt32(fileNumImg, 8);
            int columns = BitConverter.ToInt32(fileNumImg, 12);
            int size = row * columns;
            float Accuracy = 0;
            float[] trainingVal = new float[size];

            for (int q = 0; q < epoch; q++)
            {
                // Accuracy = accuracy();

                // Application.Current.Dispatcher.Invoke(() =>
                // {
                //     Alert.Content = "Accuracy: " + Accuracy + "%";
                // });

                for (int f = 0; f < nImg; f++)
                {
                    loss = 0f;

                    for (int i = 0; i < ExpVal.Length; i++)
                    {
                        ExpVal[i] = 0;
                    }

                    for (int r = 0; r < size; r++)
                    {
                        trainingVal[r] = fileImg[((f * size)) + ((r % 28) * 28 + (r / 28))] / 255f; // rotate the image becouse the EMNIST have 90 deg rotated and mirrored image
                    }

                    NeuralNetwork(trainingVal, false);

                    int index = fileLab[f];
                    ExpVal[index] = 1;

                    for (int p = 0; p < ExpVal.Length; p++)
                    {
                        loss += (float)Math.Pow(ExpVal[p] - layers[numLayers - 1][p], 2);
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

                    if (stop == true)
                    {
                        return;
                    }
                }
            }
        }

        float accuracy()
        {
            stopAccuracy = false;

            Application.Current.Dispatcher.Invoke(() =>
            {
                AccuracyAlert.Content = "Computing accuracy...";
            });

            string pathNumImg = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "emnist-digits-test-images-idx3-ubyte");
            string pathNumLab = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "emnist-digits-test-labels-idx1-ubyte");
            string pathCharImg = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "emnist-letters-test-images-idx3-ubyte");
            string pathCharLab = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "emnist-letters-test-labels-idx1-ubyte");

            byte[] fileNumImg = LittleEndianConv(File.ReadAllBytes(pathNumImg), 4);
            byte[] fileNumLab = LittleEndianConv(File.ReadAllBytes(pathNumLab), 2);
            byte[] fileCharImg = LittleEndianConv(File.ReadAllBytes(pathCharImg), 4);
            byte[] fileCharLab = LittleEndianConv(File.ReadAllBytes(pathCharLab), 2);

            byte[] cleanNumImg = removeMetadata(fileNumImg, 16);
            byte[] cleanNumLab = removeMetadata(fileNumLab, 8);
            byte[] cleanCharImg = removeMetadata(fileCharImg, 16);
            byte[] cleanCharLab = removeMetadata(fileCharLab, 8);

            for (int i = 0; i < cleanCharLab.Length; i++)
            {
                cleanCharLab[i] += 9;
            }

            byte[] fileImg = mergeFileContent(cleanNumImg, cleanCharImg);
            byte[] fileLab = mergeFileContent(cleanNumLab, cleanCharLab);

            float right = 0;
            float wrong = 0;
            float Accuracy = 0;

            int nImg = BitConverter.ToInt32(fileNumImg, 4) + BitConverter.ToInt32(fileCharImg, 4);
            int row = BitConverter.ToInt32(fileNumImg, 8);
            int columns = BitConverter.ToInt32(fileNumImg, 12);
            int size = row * columns;
            float[] trainingVal = new float[size];
            float[] ExpVal = new float[47];

            for (int i = 0; i < ExpVal.Length; i++)
            {
                ExpVal[i] = 0;
            }

            float ExpValNum = -1;

            for (int i = 0; i < nImg; i++)
            {
                for (int j = 0; j < ExpVal.Length; j++)
                {
                    ExpVal[j] = 0;
                }

                for (int j = 0; j < size; j++)
                {
                    trainingVal[j] = fileImg[((i * size)) + ((j % 28) * 28 + (j / 28))] / 255f; // rotate the image becouse the EMNIST have 90 deg rotated and mirrored image
                }

                ExpValNum = NeuralNetwork(trainingVal, false);

                int index = fileLab[i];
                ExpVal[index] = 1;

                if (ExpVal[(int)ExpValNum] == 0)
                {
                    wrong++;
                }
                else
                {
                    right++;
                }

                Accuracy = right * 100 / (right + wrong);

                if (i % 1000 == 0)
                {
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        AccuracyAlert.Content = "Accuracy: " + Accuracy + "%";
                    });
                }

                if (stopAccuracy)
                {
                    return Accuracy;
                }
            }

            Application.Current.Dispatcher.Invoke(() =>
            {
                AccuracyAlert.Content = "Loaded latest data, accuracy: " + Accuracy + "%";
            });

            return Accuracy;
        }

        byte[] LittleEndianConv(byte[] file, int numByte)
        {
            byte[] tempImg = new byte[4];
            byte[] tempLab = new byte[4];

            for (int i = 0; i < numByte; i++)
            {
                tempImg = new byte[4];
                tempLab = new byte[4];

                for (int j = 0; j < 4; j++)
                {
                    tempImg[j] = file[j + (4 * i)];
                }

                Array.Reverse(tempImg);
                Array.Reverse(tempLab);

                for (int j = 0; j < 4; j++)
                {
                    file[j + (4 * i)] = tempImg[j];
                }
            }

            return file;
        }

        byte[] removeMetadata(byte[] file, int metadataSize)
        {
            byte[] temp = new byte[file.Length - metadataSize];
            for (int i = 0; i < file.Length - metadataSize; i++)
            {
                temp[i] = file[i + metadataSize];
            }

            return temp;
        }

        byte[] mergeFileContent(byte[] file1, byte[] file2)
        {
            byte[] temp = new byte[file1.Length + file2.Length];
            
            for(int i = 0; i < file1.Length; i++)
            {
                temp[i] = file1[i];
            }

            for (int i = 0; i < file2.Length; i++)
            {
                temp[i + file1.Length] = file2[i];
            }

            return temp;
        }

        byte[] shuffle(byte[] file, int nImg, int size)
        {
            byte[] shuffledFile = new byte[file.Length];
            byte[] temp = new byte[size];
            int[] rndPos = Enumerable.Range(0, nImg).OrderBy(x => rnd.Next()).ToArray();

            for (int i = 0; i < nImg; i++)
            {
                for(int k = 0; k < size; k++)
                {
                    if (i == 0)
                    {
                        temp[k] = file[k];
                    }
                    else
                    {
                        temp[k] = file[k + i * size];
                    }
                }

                for (int k = 0; k < size; k++)
                {
                    shuffledFile[rndPos[i]] = temp[k];
                }
            }

            return shuffledFile;
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

            System.Diagnostics.Debug.WriteLine("File saved successfully");
        }

        async void loadAsync()
        {
            await Task.Run(() => load());
        }

        async void load()
        {
            string pathData = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "archive", "trainData.dat");

            try
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    Alert.Content = "Loading latest data...";
                });

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

                Application.Current.Dispatcher.Invoke(() =>
                {
                    Alert.Content = "File loaded";
                });

                await Task.Delay(100);

                stopAccuracy = true;
                await Task.Delay(500);

                float Accuracy = await Task.Run(() => accuracy());
            }
            catch
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    AccuracyAlert.Content = "Some parameter has been changed, training file reseted";
                });
                new BinaryWriter(File.Open(pathData, FileMode.Create));
            }
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

        async private void TrainButton(object sender, RoutedEventArgs e)
        {
            stop = false;
            Train.IsEnabled = false;
            Alert.Content = "Training started";
            await Task.Delay(100);
            await Task.Run(() => training());

            Train.IsEnabled = true;
            Alert.Content = "Training finished";
        }

        async private void SaveButton(object sender, RoutedEventArgs e)
        {
            stop = true;

            Alert.Content = "Saving files...";
            await Task.Delay(100);
            Save.IsEnabled = false;
            await Task.Run(() => save());
            Alert.Content = "Files saved";

            stopAccuracy = true;
            await Task.Delay(500);

            float Accuracy = await Task.Run(() => accuracy());

            Save.IsEnabled = true;
        }
    }
}
