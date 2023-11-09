using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace FaceDetection
{
    public partial class Form1 : Form
    {
        private MCvFont font = new MCvFont(Emgu.CV.CvEnum.FONT.CV_FONT_HERSHEY_TRIPLEX, 0.6d, 0.6d);

        private HaarCascade faceDetected;
        private Image<Bgr, byte> Frame;
        private Capture camera;

        private Image<Gray, byte> result;
        private Image<Gray, byte> TrainedFace = null;
        private Image<Gray, byte> grayFace = null;

        private List<Image<Gray, byte>> trainingImages = new List<Image<Gray, byte>>();
        private List<string> namesList = new List<string>();

        private string name;

        private bool isActive = false;

        public Form1()
        {
            InitializeComponent();
            faceDetected = new HaarCascade("haarcascade_frontalface_default.xml");

            try
            {
                string path = Path.Combine(Application.StartupPath, "faces/faces.txt");
                string text = File.ReadAllText(path);

                if (text.Length == 0) return;

                string[] names = text.Split(';');

                foreach (string name in names)
                {
                    string facePath = Path.Combine(Application.StartupPath, "faces", $"{name}.bmp");

                    namesList.Add(name);
                    trainingImages.Add(new Image<Gray, byte>(new Bitmap(facePath)));
                }

                namesList.Add("");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Loading error: {ex.Message}");
            }
        }

        private void start_Click(object sender, EventArgs e)
        {
            if (isActive)
            {
                isActive = false;
                toggle.Text = "Start";

                camera?.Dispose();

                return;
            }

            isActive = true;
            toggle.Text = "Stop";

            camera = new Capture();
            camera.QueryFrame();

            Application.Idle += new EventHandler(FrameProcedure);
        }

        private void saveButton_Click(object sender, EventArgs e)
        {
            grayFace = camera.QueryGrayFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            MCvAvgComp[][] DetectedFaces = grayFace.DetectHaarCascade(faceDetected, 1.2, 10, Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(20, 20));

            foreach (MCvAvgComp f in DetectedFaces[0])
            {
                TrainedFace = Frame.Copy(f.rect).Convert<Gray, byte>();
                break;
            }

            TrainedFace = result.Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);

            trainingImages.Add(TrainedFace);
            namesList.Add(textName.Text);

            foreach (var image in trainingImages)
            {
                bool isLast = trainingImages.IndexOf(image) == trainingImages.Count - 1;

                image.Save($"{Application.StartupPath}/faces/{textName.Text}.bmp");
                File.AppendAllText($"{Application.StartupPath}/faces/faces.txt", textName.Text + (isLast ? "" : ";"));
            }

            MessageBox.Show($"{textName.Text} Added Successfully");
        }

        private void FrameProcedure(object sender, EventArgs e)
        {
            if (!isActive)
            {
                cameraBox.Image = null;
                return;
            }

            Frame = camera.QueryFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            grayFace = Frame.Convert<Gray, Byte>();
            MCvAvgComp[] FacesDetectedNow = grayFace.DetectHaarCascade(faceDetected, 1.2, 10, Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(20, 20))[0];

            foreach (MCvAvgComp f in FacesDetectedNow)
            {
                result = Frame.Copy(f.rect).Convert<Gray, byte>().Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                Frame.Draw(f.rect, new Bgr(Color.Green), 3);

                if (trainingImages.Count != 0)
                {
                    MCvTermCriteria termCriteria = new MCvTermCriteria(namesList.Count, 0.001);
                    EigenObjectRecognizer recognizer = new EigenObjectRecognizer(trainingImages.ToArray(), namesList.ToArray(), 1500, ref termCriteria);

                    name = recognizer.Recognize(result);

                    Frame.Draw(name, ref font, new Point(f.rect.X - 2, f.rect.Y - 2), new Bgr(Color.Red));
                }
            }
            cameraBox.Image = Frame;
        }
    }
}