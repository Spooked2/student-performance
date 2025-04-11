//Imports
import studentData from './student_performance.json' assert {type: 'json'}

//Wait for the window to load before doing anything
window.addEventListener('load', init);

const studentTemplate = {
    Age: 15,
    Gender: 0,
    Ethnicity: 0,
    ParentalEducation: 0,
    StudyTimeWeekly: 0,
    Absences: 0,
    Tutoring: 0,
    ParentalSupport: 0,
    Extracurricular: 0,
    Sports: 0,
    Music: 0,
    Volunteering: 0,
    GPA: 0
}

//Variables
let form;
let resultP;

const nnOptions = {
    task: 'classification',
    debug: true,
    layers: [
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 64,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 64,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            activation: 'softmax',
        },
    ],
    learningRate: 0.20
}

ml5.setBackend("webgl");
const nn = ml5.neuralNetwork(nnOptions);

//Functions
async function init() {

    //Get the form element for later
    form = document.getElementById('studentForm');

    form.addEventListener("submit", submitHandler);

    resultP = document.getElementById('resultP');

    try {

        const modelDetails = {
            model: './model/studentPerformanceModel.json',
            metadata: './model/studentPerformanceModel_meta.json',
            weights: './model/studentPerformanceModel.weights.bin'
        }

        await nn.load(modelDetails, () => console.log("het model is geladen!"));

    } catch {

        const parsedStudentData = [];

        //Clean up the json data so we can use it better
        for (const student of studentData) {

            const parsedData = {
                Age: student.Age,
                Gender: student.Gender,
                Ethnicity: student.Ethnicity,
                ParentalEducation: student.ParentalEducation,
                StudyTimeWeekly: student.StudyTimeWeekly,
                Absences: student.Absences,
                Tutoring: student.Tutoring,
                ParentalSupport: student.ParentalSupport,
                Extracurricular: student.Extracurricular,
                Sports: student.Sports,
                Music: student.Music,
                Volunteering: student.Volunteering,
                GPA: student.GPA,
            }

            const label = student.GradeClass.toString();

            parsedStudentData.push({data: parsedData, label: label});
        }

        //Load json data into nn
        trainNn(parsedStudentData);

    }


}

async function trainNn(data) {

    data = data.toSorted(() => (Math.random() - 0.5));
    data = data.toSorted(() => (Math.random() - 0.5));
    data = data.toSorted(() => (Math.random() - 0.5));

    const trainingData = data.slice(0, Math.floor(data.length * 0.9))
    const testingData = data.slice(Math.floor(data.length * 0.9) + 1)

    for (const student of trainingData) {

        nn.addData(student.data, {label: student.label});

    }

    console.log('NN filled with data!');

    await nn.normalizeData();
    await nn.train({epochs: 320}, () => console.log('Finished Training!'));

    testNn(testingData);

}

async function testNn(testingData) {

    let correctAnswers = 0;

    for (const student of testingData) {

        const answer = await nn.classify(student.data);

        if (answer[0].label === student.label) {
            correctAnswers++;
        } else {
            console.log(`NN thinks a class ${student.label} student is a class ${answer[0].label}`);
        }

    }

    const accuracy = correctAnswers / testingData.length;

    console.log(`Got ${correctAnswers} correct answers out of ${testingData.length}`);
    console.log(`Accuracy of the model is about ${accuracy * 100}%`);

    nn.save("studentPerformanceModel", () => console.log("model was saved!"));
}

async function submitHandler(e) {

    e.preventDefault();

    const formData = new FormData(form);

    const formDataArray = Array.from(formData);

    let formDataObject = {};

    for (const input of formDataArray) {

        formDataObject[input[0]] = Number(input[1]);

    }

    const newStudent = {...studentTemplate, ...formDataObject};

    const result = await nn.classify(newStudent);

    let mostConfidentLabel = {confidence: 0};

    for (const label of result) {

        if (label.confidence > mostConfidentLabel.confidence) {
            mostConfidentLabel = label
        }

    }

    let grade = "no grade could be loaded!"

    switch (mostConfidentLabel.label) {
        case('0'):
            grade = 'A';
            break;
        case('1'):
            grade = 'B';
            break;
        case('2'):
            grade = 'C';
            break;
        case('3'):
            grade = 'D';
            break;
        case('4'):
            grade = 'F';
            break;
    }

    resultP.innerText = `Your grades probably average around ${grade}`;

}