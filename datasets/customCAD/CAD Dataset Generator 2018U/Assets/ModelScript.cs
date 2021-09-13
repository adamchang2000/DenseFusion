using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ModelScript : MonoBehaviour
{

    string meta_dir = "C:/Users/roar/Desktop/adam/DenseFusion/datasets/customCAD/CAD Dataset Generator 2018U/Meta/";
    string proj_txt;
    string transforms_txt;
    int counter = 0;

    // Start is called before the first frame update
    void Start()
    {
        Camera main_cam = Camera.main;

        proj_txt = meta_dir + "proj_mat.txt";
        transforms_txt = meta_dir + "transforms.txt";

        StreamWriter temp = new StreamWriter(transforms_txt);
        temp.Close();

        StreamWriter writer = new StreamWriter(proj_txt);
        Matrix4x4 projectionMatrix = GL.GetGPUProjectionMatrix(main_cam.projectionMatrix, false);
        //Matrix4x4 projectionMatrix = main_cam.projectionMatrix;
        writer.WriteLine(projectionMatrix);
        writer.Close();
    }

    // Update is called once per frame
    void Update()
    {
        int speed = 60;
        Camera main_cam = Camera.main;

        Vector3 model_pos = transform.position;

        float NEAR_CLIP = main_cam.nearClipPlane;
        float FAR_CLIP = main_cam.farClipPlane;

        Vector3 translate_vec = new Vector3(Random.Range(-100, 100), Random.Range(-100, 100), Random.Range(-100, 100)) * speed / 10;
        Vector3 rotate_vec = new Vector3(Random.Range(-180, 180), Random.Range(-180, 180), Random.Range(-180, 180));

        Debug.Log("translate vec " + translate_vec);
        transform.Translate(translate_vec);
        transform.Rotate(rotate_vec);

        int X_MAX = 1250;
        int Y_MAX = 1000;
        int CLIP_OFFSET = 1500;

        if (transform.position[0] > X_MAX || transform.position[0] < -X_MAX) {
            transform.position = new Vector3(0, transform.position[1], transform.position[2]);
        } else if (transform.position[1] > Y_MAX || transform.position[1] < -Y_MAX) {
            transform.position = new Vector3(transform.position[0], 0, transform.position[2]);
        } else if (transform.position[2] > FAR_CLIP - CLIP_OFFSET || transform.position[2] < NEAR_CLIP + CLIP_OFFSET) {
            transform.position = new Vector3(transform.position[0], transform.position[1], (NEAR_CLIP + FAR_CLIP) / 2);
        }

        Debug.Log(transform.position);
        StreamWriter writer = new StreamWriter(transforms_txt, true);
        writer.WriteLine(counter++ + "\n" + transform.localPosition + "\n" + transform.localRotation);
        writer.Close();
    }
}
