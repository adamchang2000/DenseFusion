using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;



public class CameraScript : MonoBehaviour
{   

    string meta_dir = "C:/Users/adam/CAD Dataset Generator 2018U/Meta/";
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
        Transform model_t = GameObject.Find("medical_aruco").transform;

        /*
        Vector3 model_pos = model_t.position;
        Vector3 perturb_look = new Vector3(Random.Range(-100, 100), Random.Range(-100, 100), Random.Range(-100, 100)) * speed * Time.deltaTime * 12;
        Vector3 cam_trans = Vector3.forward * Time.deltaTime * Random.Range(-100, 100) * speed * 10;

        //position the camera to look at the model and spin around it
        transform.LookAt(model_pos + perturb_look);
        //transform.LookAt(model_pos);
        transform.RotateAround(model_pos, Vector3.up, speed * Time.deltaTime * Random.Range(0, 4));
        transform.RotateAround(model_pos, Vector3.right, speed * Time.deltaTime * Random.Range(0, 4));
        transform.Translate(cam_trans);

        Camera main_cam = Camera.main;
        float NEAR_CLIP = main_cam.nearClipPlane;
        float FAR_CLIP = main_cam.farClipPlane;

        if (Vector3.Distance(transform.position, model_pos) < NEAR_CLIP) {
            transform.Translate(-cam_trans);
        }

        if (Vector3.Distance(transform.position, model_pos) > FAR_CLIP) {
            transform.Translate(-cam_trans);
        }
        */

        StreamWriter writer = new StreamWriter(transforms_txt, true);
        writer.WriteLine(counter++ + "\n" + model_t.localPosition + "\n" + model_t.localRotation);
        writer.Close();
    }
}
